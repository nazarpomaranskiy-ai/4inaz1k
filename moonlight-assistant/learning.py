"""Local patch classifier and uncertainty sampling; no pseudo-label training."""
import json
import time
import queue
import threading
from pathlib import Path
import cv2
import numpy as np
from imaging import write_image

HOG=cv2.HOGDescriptor((24,24),(12,12),(6,6),(6,6),9)

def features(image):
    image=cv2.resize(image,(24,24),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hog=HOG.compute(gray).ravel()
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist=np.concatenate([cv2.calcHist([hsv],[c],None,[16],[0,180 if c==0 else 256]).ravel()/576 for c in range(3)])
    rgb=cv2.resize(image,(4,4),interpolation=cv2.INTER_AREA).ravel()/255
    return np.concatenate([hog,hist,rgb]).astype(np.float32)

def patch_at(frame,x,y,radius):
    radius=max(4,int(radius*1.4))
    h,w=frame.shape[:2]
    x,y=round(x),round(y)
    l,t,r,b=max(0,x-radius),max(0,y-radius),min(w,x+radius+1),min(h,y+radius+1)
    return frame[t:b,l:r]

class PatchModel:
    def __init__(self,path):
        self.path=Path(path);self.model=None;self.mtime=None;self.checked=0;self.threshold=.6
        self.reload()
    def reload(self):
        now=time.monotonic()
        if now-self.checked<2:return
        self.checked=now
        if not self.path.exists():return
        mtime=self.path.stat().st_mtime_ns
        if mtime!=self.mtime:
            fs=cv2.FileStorage(self.path.read_text('utf-8'),cv2.FILE_STORAGE_READ|cv2.FILE_STORAGE_MEMORY)
            model=cv2.ml.RTrees_create();model.read(fs.getNode('opencv_ml_rtrees'));fs.release()
            self.model=model;self.mtime=mtime
            report=self.path.with_name('report.json')
            if report.exists():self.threshold=float(json.loads(report.read_text('utf-8')).get('selected_threshold',.6))
    def probabilities(self,patches):
        self.reload()
        if not patches or self.model is None:return np.zeros(len(patches),np.float32)
        values=np.stack([features(p) for p in patches])
        votes=self.model.getVotes(values,0)
        classes=votes[0];positive=np.flatnonzero(classes==1)
        if not len(positive):return np.zeros(len(patches),np.float32)
        return votes[1:,positive[0]]/np.maximum(1,votes[1:].sum(axis=1))

class ImageSink:
    def __init__(self):
        self.queue=queue.Queue(maxsize=4)
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def save(self,path,image):
        try:self.queue.put_nowait((path,image.copy()))
        except queue.Full:pass
    def run(self):
        while True:
            item=self.queue.get()
            if item is None:return
            try:write_image(*item)
            except OSError:pass
    def close(self):
        try:self.queue.put_nowait(None)
        except queue.Full:pass

class VisionEngine:
    def __init__(self,cfg,root):
        self.cfg=cfg;self.root=Path(root)
        self.model=PatchModel(self.root/'models/projectile.xml')
        self.previous=None;self.counter=0;self.sampled=0;self.last_sample=0;self.last_frame=0;self.frame_count=0;self.session=time.time_ns()
        self.pending=self.root/'learning/inbox'
        self.writer=ImageSink() if cfg.get('collect_learning',True) else None
        self.status='MODEL MISSING'
        self.unknown=[];self.overflow=False
    def detect(self,frame):
        self.counter+=1
        now=time.monotonic()
        if self.cfg.get('collect_learning',True) and self.frame_count<120 and now-self.last_frame>3:
            folder=self.root/'learning/frames';folder.mkdir(parents=True,exist_ok=True)
            if self.frame_count==0 and len(list(folder.glob('*.png')))>=120:
                self.frame_count=120
            else:
                self.writer.save(folder/f'{time.time_ns()}.png',frame)
                self.frame_count+=1;self.last_frame=now
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        contrast=cv2.absdiff(gray,cv2.GaussianBlur(gray,(0,0),3))
        # No red-only constraint: saturation, local contrast and frame change.
        color_contrast=cv2.absdiff(frame,cv2.GaussianBlur(frame,(0,0),3)).max(axis=2)
        mask=(color_contrast>32) | (contrast>24)
        if self.previous is not None and self.previous.shape==gray.shape:
            mask |= (cv2.absdiff(gray,self.previous)>35)&(contrast>12)
        self.previous=gray
        mask=mask.astype(np.uint8)*255
        l,t,r,b=self.cfg.get('arena',[0,0,frame.shape[1],frame.shape[0]])
        mask[:t]=0;mask[b:]=0;mask[:,:l]=0;mask[:,r:]=0
        for x1,y1,x2,y2 in self.cfg.get('ignore_rectangles',[]):mask[y1:y2,x1:x2]=0
        count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        h,w=gray.shape
        candidates=[]
        for i in range(1,count):
            x,y,bw,bh,area=stats[i]
            if not 3<=area<=4000 or max(bw,bh)>int(self.cfg.get('candidate_max_size',140)) or min(bw,bh)<1:continue
            cx,cy=centers[i]
            radius=max(bw,bh)/2
            candidates.append((float(cx),float(cy),float(radius)))
        # Bound inference cost, prioritizing the calibrated player vicinity.
        px,py=self.cfg['player']
        candidates.sort(key=lambda d:(d[0]-px)**2+(d[1]-py)**2)
        limit=int(self.cfg.get('max_candidates',600))
        self.overflow=len(candidates)>limit
        candidates=candidates[:limit]
        patches=[patch_at(frame,*c) for c in candidates]
        probs=self.model.probabilities(patches)
        threshold=float(self.cfg.get('bullet_confidence',self.model.threshold))
        found=[c for c,p in zip(candidates,probs) if p>=threshold]
        self.unknown=[c for c,p in zip(candidates,probs) if .3<=p<threshold]
        self.status=f'LEARNED {len(found)}/{len(candidates)}; unknown {len(self.unknown)}'
        if self.model.model is None:self.status='MODEL MISSING: train with train_detector.py'
        # Uncertainty samples are NOT assumed to be bullets or fed back as labels.
        if self.cfg.get('collect_learning',True) and self.sampled<600 and time.monotonic()-self.last_sample>.8:
            indexes=[i for i,p in enumerate(probs) if .3<=p<threshold]
            if indexes:
                self.pending.mkdir(parents=True,exist_ok=True)
                if self.sampled==0 and len(list(self.pending.glob('*.png')))>=2000:
                    self.sampled=600
                else:
                    i=indexes[self.counter%len(indexes)]
                    name=f'{time.time_ns()}.png'
                    self.writer.save(self.pending/name,patches[i])
                    (self.pending/(name+'.json')).write_text(json.dumps(dict(probability=float(probs[i]),label=None,source_group=f'{self.session}-{self.counter//60}')))
                    self.sampled+=1;self.last_sample=time.monotonic()
        return found,mask

    def close(self):
        if self.writer:self.writer.close()
