"""Replay an actual capture and compare confirmed Fire Missile tracks to labels."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from realm_pilot import RealmPilot,confirmed_shots,merge_pilot_motion
from tracking import CameraMotion,AdaptiveTracker
from imaging import read_image
from planner import Shot,recommend
from autododge import choose_movement


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,default=Path('pilot_validation.json'))
    args=p.parse_args();cv2.setNumThreads(2)
    rows=[json.loads(line) for line in (args.capture/'frames.jsonl').read_text().splitlines()]
    annotations=json.loads(args.annotations.read_text('utf-8'));labels={int(k):v for k,v in annotations['frames'].items()}
    cfg=json.loads((args.capture/'calibration.json').read_text('utf-8'))
    l,t,r,b=cfg['arena'];cfg['arena']=[0,0,r-l,b-t];cfg['player']=[cfg['player'][0]-l,cfg['player'][1]-t]
    cfg['ignore_rectangles']=[]
    segments=[]
    for index in sorted(labels):
        first=max(0,index-25)
        if segments and first<=segments[-1][1]+1:segments[-1][1]=max(index,segments[-1][1])
        else:segments.append([first,index])
    tp=fp=fn=0;times=[];reports=[];commands=0
    for first,last in segments:
        engine=RealmPilot(cfg,args.root);camera=CameraMotion(cfg['player']);tracker=AdaptiveTracker(gate=80,retain_seconds=.25);previous=None
        for index in range(first,last+1):
            row=rows[index];frame=read_image(args.capture/row['image'])
            if frame is None:raise OSError(row['image'])
            started=time.perf_counter();offset=camera.update(frame)
            dt=row['time']-previous if previous is not None else 1/15;previous=row['time']
            hints=[(*tuple(track.predict(tracker.stamp+dt)-offset),track.radius) for track in tracker.tracks if tracker.stamp-track.history[-1][0]<.2]
            found,_=engine.detect(frame,shift=camera.shift,camera_confidence=camera.confidence,hints=hints,dt=dt)
            shots=tracker.update([(x+offset[0],y+offset[1],rad) for x,y,rad in found],dt)
            shots=merge_pilot_motion(tracker,shots,engine,offset)
            local=[Shot(s.x-offset[0],s.y-offset[1],s.vx,s.vy,s.radius,s.ax,s.ay,s.uncertainty) for s in shots]
            best,scores=recommend(cfg['player'],local,cfg['arena'],cfg['player_speed'],cfg['player_radius'],cfg['horizon'],cfg['margin'])
            observation=dict(stamp=row['time'],decision_stamp=row['time'],best=best,scores=scores,tracks=len(shots))
            if choose_movement(observation,row['time']):commands+=1
            times.append((time.perf_counter()-started)*1000)
            if index not in labels:continue
            predicted=[(float(s.x),float(s.y)) for s in local] if engine.scene_supported else []
            targets=labels[index]['targets'];pairs=[]
            for i,target in enumerate(targets):
                for j,pred in enumerate(predicted):
                    distance=float(np.linalg.norm(np.array(target[:2])-pred))
                    if distance<=18:pairs.append((distance,i,j))
            matched_i=set();matched_j=set()
            for _,i,j in sorted(pairs):
                if i not in matched_i and j not in matched_j:matched_i.add(i);matched_j.add(j)
            ft=len(matched_i);ff=len(predicted)-ft;fm=len(targets)-ft;tp+=ft;fp+=ff;fn+=fm
            reports.append(dict(index=index,kind=labels[index].get('kind'),targets=targets,predictions=predicted,
                                tp=ft,fp=ff,fn=fm,scene_supported=engine.scene_supported,appearance=engine.candidates,moving=found))
    report=dict(detector='realm-fire-missile-pilot',scope='Fire Missile only; labeled replay frames, not live dodge success',
                capture=str(args.capture),labeled_frames=len(reports),targets=tp+fn,true_positive=tp,
                false_positive=fp,false_negative=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                replay_commands=commands,production_ready=False,frames=reports)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)

if __name__=='__main__':main()
