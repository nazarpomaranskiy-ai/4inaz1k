"""Evaluate a one-class network only on real frames excluded from training."""
import argparse
import json
from pathlib import Path
import time
import hashlib
import cv2
import numpy as np


def match_boxes(targets,predictions):
    pairs=[]
    for i,box in enumerate(targets):
        a=np.array(box,float)
        for j,pred in enumerate(predictions):
            b=np.array(pred[:4],float)
            intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
            iou=float(intersection/max(1e-9,union))
            if iou>=.3:pairs.append((-iou,i,j))
    used_a=set();used_b=set()
    for _,i,j in sorted(pairs):
        if i not in used_a and j not in used_b:used_a.add(i);used_b.add(j)
    return len(used_a),len(predictions)-len(used_b),len(targets)-len(used_a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('weights',type=Path);p.add_argument('dataset',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--confidence',type=float,default=.5)
    p.add_argument('--image-size',type=int,default=1600,
                   help='Inference resolution; small projectiles disappear when the viewport is downscaled')
    args=p.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2);cv2.setNumThreads(2)
    model=YOLO(str(args.weights));device=0 if torch.cuda.is_available() else 'cpu'
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    rows=[];tp=fp=fn=0;timings=[]
    for item in manifest['real_holdout']:
        frame=cv2.imread(item['image'])
        if frame is None:raise ValueError(f"Cannot read evaluation image: {item['image']}")
        for _ in range(2):model.predict(frame,imgsz=args.image_size,device=device,conf=args.confidence,verbose=False)
        started=time.perf_counter()
        result=model.predict(frame,imgsz=args.image_size,device=device,conf=args.confidence,iou=.45,verbose=False)[0]
        predictions=result.boxes.data.detach().cpu().numpy().tolist()
        predictions=[v for v in predictions if min(v[2]-v[0],v[3]-v[1])>=2 and max(v[2]-v[0],v[3]-v[1])<=80]
        timings.append((time.perf_counter()-started)*1000)
        a,b,c=match_boxes(item['boxes'],predictions);tp+=a;fp+=b;fn+=c
        rows.append(dict(index=item['index'],targets=item['boxes'],predictions=predictions,tp=a,fp=b,fn=c))
    report=dict(detector='neural-fire-missile',scope='One projectile family; real frames held out from training, from one recording only',
                confidence=args.confidence,image_size=args.image_size,
                weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                targets=tp+fn,true_positive=tp,false_positive=fp,false_negative=fn,
                precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),labeled_frames=len(rows),
                median_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)),
                production_ready=False,frames=rows)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
