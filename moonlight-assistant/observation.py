"""Bounded passive game observation. No input injection or automatic training."""
import json
from pathlib import Path
import queue
import random
import shutil
import threading
import time
import cv2
import numpy as np
from tracking import CameraMotion, AdaptiveTracker


def atomic_json(path, value):
    path=Path(path);temp=path.with_suffix('.pending.json')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


class Budget:
    def __init__(self, root, max_mb=768, total_mb=2048, reserve_mb=1024):
        if min(max_mb,total_mb,reserve_mb)<0 or min(max_mb,total_mb)==0:
            raise ValueError('Storage limits must be positive and reserve nonnegative')
        self.root=Path(root);self.limit=int(max_mb*1024**2);self.reserve=int(reserve_mb*1024**2)
        existing=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        self.allowance=min(self.limit,max(0,int(total_mb*1024**2)-existing));self.used=0
        self.lock=threading.Lock()
    def reason(self, incoming=0, free=None):
        if self.used+incoming>self.allowance:return 'storage_limit'
        free=shutil.disk_usage(self.root).free if free is None else free
        if free-incoming<self.reserve:return 'low_disk_space'
        return ''
    def claim(self, count):
        with self.lock:
            reason=self.reason(count)
            if not reason:self.used+=count
            return reason


class Analyzer:
    """A small latest-frame queue keeps analysis from stalling the recorder."""
    def __init__(self, out, cfg, detector_factory, review_limit=48, budget=None):
        self.out=Path(out);self.cfg=cfg;self.factory=detector_factory;self.review_limit=review_limit
        self.pending=queue.Queue(maxsize=2);self.done=threading.Event();self.thread=None
        self.dropped=0;self.analyzed=0;self.detections=0;self.review=[];self.error=''
        self.prediction_errors=[];self.latencies=[];self.categories={};self.previous_detections=0
        self.budget=budget
        self.detector_metadata={}
        self.review_seen=0;self.rng=random.Random(42)
    def start(self):
        self.thread=threading.Thread(target=self.run,name='passive-analysis',daemon=True);self.thread.start()
    def submit(self, name, frame, stamp):
        if self.error:return
        try:self.pending.put_nowait((name,frame,stamp))
        except queue.Full:
            try:self.pending.get_nowait();self.dropped+=1
            except queue.Empty:pass
            try:self.pending.put_nowait((name,frame,stamp))
            except queue.Full:self.dropped+=1
    def finish(self):
        self.done.set()
        if self.thread:self.thread.join()
    def run(self):
        detector=None
        try:
            detector=self.factory()
            self.detector_metadata=dict(weights_sha256=getattr(detector,'weights_sha256',None),
                                        confidence=getattr(detector,'threshold',None),
                                        image_size=getattr(detector,'image_size',None))
            l,t,_,_=self.cfg['arena'];px,py=self.cfg['player']
            camera=CameraMotion([px-l,py-t]);tracker=AdaptiveTracker(gate=80,retain_seconds=.25)
            previous=None
            with (self.out/'observations.jsonl').open('ab') as log:
                while not self.done.is_set() or not self.pending.empty():
                    try:name,frame,stamp=self.pending.get(timeout=.1)
                    except queue.Empty:continue
                    started=time.perf_counter();dt=0 if previous is None else stamp-previous
                    # A gap breaks correspondence; do not fit trajectories across it.
                    if previous is None or dt>.25 or dt<=0:
                        camera=CameraMotion([px-l,py-t]);tracker=AdaptiveTracker(gate=80,retain_seconds=.25)
                    offset=camera.update(frame);found,_=detector.detect(frame)
                    supported=bool(detector.scene_supported)
                    if not supported or camera.rotated:
                        tracker=AdaptiveTracker(gate=80,retain_seconds=.25)
                        shots=[]
                    else:shots=tracker.update([(x+offset[0],y+offset[1],r) for x,y,r in found],max(.001,dt))
                    previous=stamp
                    errors=[float(track.errors[track.model]) for track in tracker.tracks if len(track.history)>=3 and track.missed==0]
                    latency=(time.perf_counter()-started)*1000
                    self.latencies.append(latency);self.prediction_errors.extend(errors)
                    self.analyzed+=1;self.detections+=len(found)
                    boxes=getattr(detector,'boxes',[])
                    reasons=[]
                    if any(len(b)>=5 and float(b[4])<.65 for b in boxes):reasons.append('low_confidence')
                    if errors and max(errors)>12:reasons.append('trajectory_error')
                    if camera.rotated:reasons.append('camera_rotation_or_zoom')
                    if len(found)!=self.previous_detections:reasons.append('detection_count_changed')
                    if self.analyzed%150==1:reasons.append('periodic_sample')
                    self.previous_detections=len(found)
                    row=dict(image=name,monotonic=stamp,dt=dt,scene_supported=supported,
                             boxes=boxes,labels_reviewed=False,camera_offset=offset.tolist(),
                             camera_confidence=camera.confidence,camera_rotated=camera.rotated,
                             player_world_estimate=[px-l+float(offset[0]),py-t+float(offset[1])],
                             tracks=[dict(identity=tr.identity,model=tr.model,observations=len(tr.history),
                                          prediction_error_px=tr.errors[tr.model],missed=tr.missed,
                                          recent_world_positions=list(tr.history)) for tr in tracker.tracks],
                             shots=[vars(s) for s in shots],analysis_ms=latency,review_reasons=reasons)
                    data=(json.dumps(row)+'\n').encode()
                    if self.budget:
                        reason=self.budget.claim(len(data))
                        if reason:raise RuntimeError(reason)
                    log.write(data);log.flush()
                    # Spread examples through the session; detector output is never a ground-truth label.
                    category=reasons[0] if reasons else ''
                    last=self.categories.get(category,-1e9)
                    if category and self.review_limit>0 and stamp-last>=5:
                        self.categories[category]=stamp
                        item=dict(image=name,reasons=reasons,labels_reviewed=False)
                        self.review_seen+=1
                        if len(self.review)<self.review_limit:self.review.append(item)
                        else:
                            index=self.rng.randrange(self.review_seen)
                            if index<self.review_limit:self.review[index]=item
        except Exception as exc:self.error=f'{type(exc).__name__}: {exc}'
        finally:
            if detector is not None:detector.close()

    def report(self):
        return dict(analyzed_frames=self.analyzed,dropped_analysis_frames=self.dropped,
                    detections=self.detections,error=self.error,review_samples=len(self.review),
                    median_analysis_ms=float(np.median(self.latencies)) if self.latencies else None,
                    prediction_error_px_median=float(np.median(self.prediction_errors)) if self.prediction_errors else None,
                    scope='Fire Missile only; detector hypotheses, not ground truth',
                    detector=self.detector_metadata,
                    prediction_error_definition='Tracker consistency against its next detection; not ground-truth accuracy',
                    automatic_training=False,production_ready=False)


def write_review(out, samples, budget=None):
    """Bounded contact sheets; the full original frames remain available for annotation."""
    out=Path(out);review=out/'review';review.mkdir(exist_ok=True)
    samples=sorted(samples,key=lambda item:item['image'])
    atomic_json(review/'queue.json',dict(labels_reviewed=False,samples=samples))
    from imaging import read_image
    for page in range(0,len(samples),12):
        group=samples[page:page+12];sheet=np.zeros((len(range(0,len(group),3))*190,960,3),np.uint8)
        for i,item in enumerate(group):
            image=read_image(out/item['image'])
            if image is None:continue
            thumb=cv2.resize(image,(320,148));x=(i%3)*320;y=(i//3)*190
            sheet[y:y+148,x:x+320]=thumb
            cv2.putText(sheet,item['image'],(x+4,y+163),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.putText(sheet,item['reasons'][0],(x+4,y+180),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,220,255),1)
        ok,encoded=cv2.imencode('.jpg',sheet)
        if ok and (budget is None or not budget.claim(len(encoded))):
            (review/f'contact_{page//12:02d}.jpg').write_bytes(encoded.tobytes())
