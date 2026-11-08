"""Bounded game-only combat recording. Never imports or sends keyboard input."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import cv2
from gamecapture import WindowsGameCapture
from imaging import write_image

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--fps',type=float,default=15)
    p.add_argument('--max-mb',type=int,default=256)
    args=p.parse_args()
    cv2.setNumThreads(2)
    cfg=json.loads((HERE/'live_config.json').read_text('utf-8'))
    base=Path(os.environ.get('LOCALAPPDATA',str(HERE)))/'MoonlightAssistant/captures'
    out=base/datetime.now().strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True,exist_ok=False)
    (out/'calibration.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    state_path=HERE/'capture_status.json'
    count=0;size=0;last=0;report=0;start=time.perf_counter()
    def status(state):
        data=dict(state=state,path=str(out),frames=count,bytes=size,elapsed=time.perf_counter()-start,
                  game_input=False)
        temp=state_path.with_suffix('.pending.json');temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
        os.replace(temp,state_path)
    try:
        with (out/'frames.jsonl').open('w',encoding='utf-8') as log,WindowsGameCapture(cfg['monitor'],cfg['arena']) as cap:
            status('recording');print('RECORDING GAME ONLY:',out,flush=True)
            while time.perf_counter()-start<args.seconds and size<args.max_mb*1024**2:
                if (out/'stop').exists():break
                sample=cap.read()
                if sample is None:time.sleep(.003);continue
                image,stamp=sample
                if stamp-last<1/args.fps:continue
                last=stamp
                ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,75])
                if not ok:raise OSError('JPEG encoding failed')
                name=f'{count:06d}.jpg';data=encoded.tobytes();(out/name).write_bytes(data)
                log.write(json.dumps(dict(image=name,time=stamp-start,monotonic=stamp))+'\n')
                count+=1;size+=len(data)
                if time.perf_counter()-report>2:
                    log.flush();status('recording');report=time.perf_counter()
            status('complete');print(f'SAVED {count} frames, {size/1024**2:.1f} MiB',flush=True)
    except Exception as exc:
        status('error: '+str(exc));raise

if __name__=='__main__':main()
