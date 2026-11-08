"""GPU training for one projectile family, isolated from live game control."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=25)
    args=p.parse_args()
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Expected CUDA GPU; refusing an accidental long CPU training run')
    args.output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    model=YOLO('yolo11n.pt')
    def report(trainer):
        metrics={k:float(v) for k,v in (trainer.metrics or {}).items()}
        value=dict(epoch=trainer.epoch+1,epochs=args.epochs,elapsed=time.perf_counter()-started,
                   metrics=metrics,save_dir=str(trainer.save_dir))
        temp=args.output/'training_status.pending.json';temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
        temp.replace(args.output/'training_status.json')
    model.add_callback('on_fit_epoch_end',report)
    model.train(data=str(args.dataset/'dataset.yaml'),project=str(args.output),name='fire_missile',
                epochs=args.epochs,imgsz=640,batch=4,device=0,workers=0,amp=False,
                # The dataset already has one class; single_cls=True renames it to "item".
                freeze=10,cache=False,seed=42,deterministic=True,single_cls=False,
                degrees=180,translate=.1,scale=.35,fliplr=.5,flipud=.5,
                mosaic=.5,close_mosaic=5,mixup=0,plots=False,val=False,patience=10)
    print('Training finished; real holdout evaluation is still required.',flush=True)

if __name__=='__main__':main()
