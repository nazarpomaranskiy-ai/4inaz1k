"""Observe a bounded play session; never sends keys, changes weights or labels predictions as truth."""
import argparse
from datetime import datetime
import ctypes
import json
import os
from pathlib import Path
import time
import cv2
from gamecapture import WindowsGameCapture
from observation import Analyzer, Budget, atomic_json, write_review

HERE=Path(__file__).resolve().parent


class GameKeys:
    """Read only WASD/F9 and only when RotMGExalt owns the foreground window."""
    def __init__(self):
        from ctypes import wintypes
        self.api=ctypes.WinDLL('user32',use_last_error=True)
        self.api.GetForegroundWindow.restype=wintypes.HWND
        self.api.GetWindowTextW.argtypes=[wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
        self.api.GetAsyncKeyState.argtypes=[ctypes.c_int]
        self.api.GetAsyncKeyState.restype=ctypes.c_short
    def read(self):
        title=ctypes.create_unicode_buffer(256)
        self.api.GetWindowTextW(self.api.GetForegroundWindow(),title,len(title))
        active=title.value.strip().lower()=='rotmgexalt'
        return dict(game_foreground=active,
                    wasd=[key for key in 'WASD' if active and self.api.GetAsyncKeyState(ord(key))&0x8000],
                    stop=bool(active and self.api.GetAsyncKeyState(0x78)&0x8000))


def record_session(cfg, base, seconds=600, fps=15, max_mb=768, total_mb=2048,
                   reserve_mb=1024, detector_factory=None, capture_factory=WindowsGameCapture,
                   key_reader=None, state_path=None):
    if not 0<seconds<=3600:raise ValueError('Duration must be between 0 and 3600 seconds')
    if not 1<=fps<=30:raise ValueError('FPS must be between 1 and 30')
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    budget=Budget(base,max_mb,total_mb,reserve_mb)
    reason=budget.reason(1)
    if reason:raise RuntimeError('Cannot start observation: '+reason)
    out=base/datetime.now().strftime('%Y%m%d-%H%M%S-%f');out.mkdir()
    atomic_json(out/'calibration.json',cfg)
    atomic_json(out/'session.json',dict(version=1,created=datetime.now().astimezone().isoformat(),
                requested_seconds=seconds,requested_fps=fps,max_mb=max_mb,total_mb=total_mb,
                game_input=False,keyboard_scope='WASD states only while RotMGExalt is foreground; sampled, not a keylogger',
                predictions_are_labels=False,automatic_training=False))
    analyzer=Analyzer(out,cfg,detector_factory,budget=budget) if detector_factory else None
    count=0;last=None;start=time.perf_counter();last_frame=start;last_status=0;failure='';reason='duration'
    def status(state):
        value=dict(state=state,path=str(out),frames=count,bytes=budget.used,
                   elapsed=time.perf_counter()-start,game_input=False,
                   analysis=analyzer.report() if analyzer else None)
        atomic_json(out/'status.json',value)
        if state_path:atomic_json(state_path,value)
    try:
        if analyzer:analyzer.start()
        print('PASSIVE OBSERVATION: play manually. F9 in game stops and saves. No WASD will be sent.',flush=True)
        print('SESSION:',out,flush=True)
        with (out/'frames.jsonl').open('ab') as log,capture_factory(cfg['monitor'],cfg['arena']) as cap:
            status('recording')
            while time.perf_counter()-start<seconds:
                now=time.perf_counter()
                if (out/'stop').exists():reason='stop_file';break
                keys=key_reader.read() if key_reader else dict(game_foreground=None,wasd=[],stop=False)
                control_stamp=time.perf_counter()
                if keys['stop']:reason='F9';break
                reason_now=budget.reason(1)
                if reason_now:reason=reason_now;break
                sample=cap.read()
                if sample is None:
                    if now-last_frame>15:reason='no_game_frames';break
                    time.sleep(.003);continue
                image,stamp=sample
                if now-stamp>.5:continue
                last_frame=now
                if last is not None and stamp-last<1/fps:continue
                last=stamp
                ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,75])
                if not ok:raise OSError('JPEG encoding failed')
                name=f'{count:06d}.jpg';data=encoded.tobytes()
                row=(json.dumps(dict(image=name,time=stamp-start,monotonic=stamp,
                                     control_sample_monotonic=control_stamp,
                                     game_foreground=keys['game_foreground'],wasd=keys['wasd']))+'\n').encode()
                reason_now=budget.claim(len(data)+len(row))
                if reason_now:reason=reason_now;break
                (out/name).write_bytes(data);log.write(row);log.flush();count+=1
                if analyzer:analyzer.submit(name,image,stamp)
                if now-last_status>=2:status('recording');last_status=now
    except KeyboardInterrupt:reason='interrupted'
    except Exception as exc:reason='error';failure=f'{type(exc).__name__}: {exc}'
    finally:
        recording_elapsed=time.perf_counter()-start
        status('finishing')
        if analyzer:analyzer.finish()
        if analyzer:write_review(out,analyzer.review,budget)
        report=dict(state='error' if failure else 'complete',stop_reason=reason,error=failure,
                    path=str(out),frames=count,recording_seconds=recording_elapsed,
                    average_recording_fps=count/max(.001,recording_elapsed),budgeted_bytes=budget.used,
                    game_input=False,wasd_sampling='Separate frame/control monotonic timestamps; short taps may be missed',
                    analysis=analyzer.report() if analyzer else None,
                    next_steps=['Review full frames including missed projectiles; thumbnails are only a starting point',
                                'Manually label projectiles and hard negatives; keep predictions unreviewed until checked',
                                'Split by independent play sessions before training; evaluate trajectories separately'],
                    automatic_training=False,production_ready=False)
        atomic_json(out/'report.json',report)
        status('error' if failure else 'complete')
        print('SAVED:',out/'report.json',flush=True)
        if analyzer and analyzer.error:print('Analysis incomplete:',analyzer.error,flush=True)
    if failure:raise RuntimeError(failure)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=600);p.add_argument('--fps',type=float,default=15)
    p.add_argument('--max-mb',type=int,default=768);p.add_argument('--total-mb',type=int,default=2048)
    p.add_argument('--reserve-mb',type=int,default=1024);p.add_argument('--output',type=Path)
    p.add_argument('--no-analysis',action='store_true',help='Record frames and manual controls without loading a detector')
    args=p.parse_args();cv2.setNumThreads(2)
    cfg=json.loads((HERE/'live_config.json').read_text('utf-8-sig'))
    runtime=HERE/'neural_runtime.json'
    runtime=json.loads(runtime.read_text('utf-8-sig')) if runtime.exists() else {}
    base=args.output or Path(runtime.get('observations_dir',str(Path(os.environ.get('LOCALAPPDATA',str(HERE)))/'MoonlightAssistant/observations')))
    def factory():
        from neural_detector import NeuralFireDetector
        local=dict(cfg);l,t,r,b=cfg['arena'];local['arena']=[0,0,r-l,b-t]
        return NeuralFireDetector(local,HERE)
    record_session(cfg,base,args.seconds,args.fps,args.max_mb,args.total_mb,args.reserve_mb,
                   None if args.no_analysis else factory,key_reader=GameKeys(),state_path=HERE/'observation_status.json')

if __name__=='__main__':main()
