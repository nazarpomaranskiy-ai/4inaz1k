"""Windows live advisory overlay. Never sends game input."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk

import cv2
import mss
import numpy as np
from PIL import Image, ImageTk
from analyze import Tracker, detect
from planner import recommend

HERE = Path(__file__).resolve().parent
KEY = '#010203'


def local_config(cfg):
    """Convert monitor-relative calibration to capture-relative coordinates."""
    c = dict(cfg)
    l, t, r, b = cfg['arena']
    c['arena'] = [0, 0, r-l, b-t]
    c['player'] = [cfg['player'][0]-l, cfg['player'][1]-t]
    c['ignore_rectangles'] = [[max(0,x1-l), max(0,y1-t), min(r-l,x2-l), min(b-t,y2-t)]
                              for x1,y1,x2,y2 in cfg.get('ignore_rectangles', [])
                              if x2 > l and y2 > t and x1 < r and y1 < b]
    return c


def calibrate(monitor, path):
    with mss.mss() as capture:
        image = Image.frombytes('RGB', (monitor['width'],monitor['height']),
                                capture.grab(monitor).rgb)
    cfg = json.loads((path if path.exists() else HERE/'config.json').read_text('utf-8'))
    root = tk.Tk()
    root.title('Select arena, then click player hitbox. Esc cancels.')
    scale = min(1, 1200/image.width, 760/image.height)
    preview = ImageTk.PhotoImage(image.resize((round(image.width*scale),round(image.height*scale))))
    canvas = tk.Canvas(root,width=preview.width(),height=preview.height(),highlightthickness=0)
    canvas.pack()
    canvas.create_image(0,0,anchor='nw',image=preview)
    state = {'start': None, 'rect': None, 'arena': None}
    def down(e):
        if state['arena']:
            px, py = round(e.x/scale), round(e.y/scale)
            l,t,r,b = state['arena']
            if not (l < px < r and t < py < b):
                return
            cfg.update(frame_size=[image.width,image.height],arena=state['arena'],
                       player=[px,py],ignore_rectangles=[],monitor=monitor)
            path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            root.destroy()
            print(f'Calibration saved: {path}')
        else:
            state['start'] = (e.x,e.y)
            if state['rect']:
                canvas.delete(state['rect'])
            state['rect'] = canvas.create_rectangle(e.x,e.y,e.x,e.y,outline='yellow',width=2)
    def drag(e):
        if state['start'] and not state['arena']:
            canvas.coords(state['rect'],*state['start'],e.x,e.y)
    def up(e):
        if not state['start'] or state['arena']:
            return
        x,y = state['start']
        l,r = sorted([round(x/scale),round(e.x/scale)])
        t,b = sorted([round(y/scale),round(e.y/scale)])
        l,t,r,b = max(0,l),max(0,t),min(image.width,r),min(image.height,b)
        if r-l >= 100 and b-t >= 100:
            state['arena'] = [l,t,r,b]
            root.title('Now click the center of the player hitbox')
    canvas.bind('<ButtonPress-1>',down)
    canvas.bind('<B1-Motion>',drag)
    canvas.bind('<ButtonRelease-1>',up)
    root.bind('<Escape>',lambda e:root.destroy())
    root.mainloop()


def configure_window(root):
    """Click-through, no activation, and exclusion from desktop capture."""
    root.update_idletasks()
    user32 = ctypes.WinDLL('user32',use_last_error=True)
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND
    hwnd = user32.GetParent(root.winfo_id())
    get_style = user32.GetWindowLongW
    set_style = user32.SetWindowLongW
    get_style.argtypes = [wintypes.HWND,ctypes.c_int]
    get_style.restype = ctypes.c_long
    set_style.argtypes = [wintypes.HWND,ctypes.c_int,ctypes.c_long]
    set_style.restype = ctypes.c_long
    style = get_style(hwnd,-20)
    set_style(hwnd,-20,style | 0x80000 | 0x20 | 0x08000000 | 0x80)
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND,wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    if sys.getwindowsversion().build < 19041:
        raise RuntimeError('Capture exclusion requires Windows 10 build 19041 or newer.')
    if not user32.SetWindowDisplayAffinity(hwnd,0x11):
        raise ctypes.WinError(ctypes.get_last_error())
    return user32


def worker(cfg, monitor, messages, stop, paused, record):
    def publish(value):
        try:
            messages.get_nowait()
        except queue.Empty:
            pass
        messages.put_nowait(value)
    c = local_config(cfg)
    l,t,r,b = cfg['arena']
    region = dict(left=monitor['left']+l,top=monitor['top']+t,width=r-l,height=b-t)
    tracker = Tracker(c['tracking_gate'],c['max_shot_speed'])
    log = None
    try:
        if record:
            record.mkdir(parents=True,exist_ok=False)
            (record/'calibration.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            log = (record/'observations.jsonl').open('w',encoding='utf-8')
        with mss.mss() as capture:
            previous = None
            last_save = 0
            index = 0
            while not stop.is_set():
                if paused.is_set():
                    previous = None
                    tracker.tracks = []
                    stop.wait(.05)
                    continue
                started = time.perf_counter()
                frame = np.asarray(capture.grab(region))[:,:,:3].copy()
                stamp = time.perf_counter()
                dt = stamp-previous if previous is not None else 1/30
                previous = stamp
                if dt > .2:
                    tracker.tracks = []
                detections,_ = detect(frame,c)
                shots = tracker.update(detections,max(dt,.001))
                best, scores = recommend(c['player'],shots,c['arena'],c['player_speed'],
                                         c['player_radius'],c['horizon'],c['margin'])
                result = dict(stamp=stamp,best=best,scores=scores,detections=detections,
                              tracks=len(shots),ms=(time.perf_counter()-started)*1000)
                publish(result)
                if log and stamp-last_save >= .2:
                    name = f'{index:07d}.jpg'
                    if not cv2.imwrite(str(record/name),frame):
                        raise RuntimeError('Cannot write observation image')
                    log.write(json.dumps(dict(image=name,monotonic=stamp,best=best,
                                              tracks=[vars(s) for s in shots],
                                              label=None))+'\n')
                    log.flush()
                    last_save,index = stamp,index+1
                stop.wait(max(0,1/30-(time.perf_counter()-started)))
    except Exception as exc:
        publish({'error':str(exc),'stamp':time.perf_counter()})
    finally:
        if log:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibrate',action='store_true')
    p.add_argument('--monitor',type=int,default=1)
    p.add_argument('--config',type=Path,default=HERE/'live_config.json')
    p.add_argument('--record',action='store_true',help='Save arena screenshots + unlabeled observations')
    args = p.parse_args()
    if sys.platform != 'win32':
        p.error('Live overlay requires Windows; use analyze.py for offline videos.')
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        ctypes.windll.user32.SetProcessDPIAware()
    with mss.mss() as capture:
        if args.monitor < 1 or args.monitor >= len(capture.monitors):
            p.error('Invalid monitor number')
        monitor = dict(capture.monitors[args.monitor])
    if args.calibrate:
        calibrate(monitor,args.config)
        return
    if not args.config.exists():
        p.error('Run overlay.py --calibrate first.')
    cfg = json.loads(args.config.read_text('utf-8'))
    if cfg.get('monitor') != monitor:
        p.error('Monitor geometry changed. Calibrate again.')
    l,t,r,b = cfg['arena']
    px,py = cfg['player']
    if not (0<=l<px<r<=monitor['width'] and 0<=t<py<b<=monitor['height']):
        p.error('Invalid arena/player calibration.')
    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{monitor['width']}x{monitor['height']}{monitor['left']:+d}{monitor['top']:+d}")
    root.configure(bg=KEY)
    root.attributes('-topmost',True)
    root.attributes('-transparentcolor',KEY)
    canvas = tk.Canvas(root,bg=KEY,highlightthickness=0)
    canvas.pack(fill='both',expand=True)
    try:
        user32 = configure_window(root)
    except Exception:
        root.destroy()
        raise
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    stop,paused = threading.Event(),threading.Event()
    messages = queue.Queue(maxsize=1)
    record = HERE/'sessions'/datetime.now().strftime('%Y%m%d-%H%M%S-%f') if args.record else None
    thread = threading.Thread(target=worker,args=(cfg,monitor,messages,stop,paused,record),daemon=True)
    thread.start()
    latest = None
    keys = {0x77:False,0x78:False}
    def close():
        stop.set()
        root.destroy()
    def tick():
        nonlocal latest
        for key in keys:
            pressed = bool(user32.GetAsyncKeyState(key)&0x8000)
            if pressed and not keys[key]:
                if key==0x78:
                    close()
                    return
                if paused.is_set():
                    paused.clear()
                    latest = None
                else:
                    paused.set()
            keys[key] = pressed
        try:
            latest = messages.get_nowait()
        except queue.Empty:
            pass
        canvas.delete('all')
        message = 'WAITING FOR TRACKS'
        color = '#ffcc44'
        valid = latest and time.perf_counter()-latest['stamp'] < .2
        if paused.is_set():
            message = 'PAUSED'
        elif latest and 'error' in latest:
            message = 'ERROR: '+latest['error']
        elif valid and latest['best']:
            best = latest['best']
            name,gap,vx,vy = best
            color = '#44ff88' if gap>0 else '#ff5555'
            message = f'{name} | clearance {gap:.0f}px | {latest["tracks"]} tracks | {latest["ms"]:.0f}ms'
            if gap<=0:
                message = 'NO SAFE CANDIDATE | '+message
            # Candidate rays are short forecasts, not verified safe corridors.
            for _,clearance,ux,uy in latest['scores']:
                canvas.create_line(px,py,px+ux*cfg['horizon'],py+uy*cfg['horizon'],
                                   fill='#448866' if clearance>0 else '#883333',width=2)
            canvas.create_line(px,py,px+vx*cfg['horizon'],py+vy*cfg['horizon'],
                               fill=color,width=5,arrow=tk.LAST)
        elif latest and not valid:
            message = 'STALE CAPTURE: recommendation hidden'
        if valid and not paused.is_set() and 'detections' in latest:
            for x,y,radius in latest['detections']:
                x,y = x+l,y+t
                canvas.create_oval(x-radius,y-radius,x+radius,y+radius,outline='#ffcc44')
        canvas.create_oval(px-7,py-7,px+7,py+7,outline='white')
        canvas.create_rectangle(12,12, min(monitor['width']-12,920),82,fill='#181818',outline='')
        canvas.create_text(22,28,text=message,fill=color,anchor='w',font=('Segoe UI',12,'bold'))
        canvas.create_text(22,57,text='EXPERIMENTAL | F8 pause | F9 exit | fixed player/camera | '+
                           ('RECORDING ARENA' if record else 'no recording'),
                           fill='white',anchor='w',font=('Segoe UI',10))
        root.after(33,tick)
    root.protocol('WM_DELETE_WINDOW',close)
    try:
        tick()
        root.mainloop()
    finally:
        stop.set()
        thread.join(timeout=2)

if __name__ == '__main__':
    main()
