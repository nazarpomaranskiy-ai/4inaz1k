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


def focus_game():
    """Bring the exact RotMG game window forward; never send game input."""
    if sys.platform != 'win32':
        return False
    user32 = ctypes.WinDLL('user32',use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowTextW.argtypes = [wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND,ctypes.c_int]
    def title(hwnd):
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd,buf,512)
        return buf.value
    windows = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    @callback_type
    def collect(hwnd, unused):
        if title(hwnd).replace(' ','').lower() == 'rotmgexalt':
            windows.append(hwnd)
        return True
    user32.EnumWindows(collect,0)
    if not windows:
        return False
    user32.ShowWindow(windows[0],9)
    user32.SetForegroundWindow(windows[0])
    return title(user32.GetForegroundWindow()).replace(' ','').lower() == 'rotmgexalt'


def calibrate(monitor, path):
    cfg = json.loads((path if path.exists() else HERE/'config.json').read_text('utf-8'))
    root = tk.Tk()
    root.title('Moonlight calibration: capturing game in 5 seconds')
    scale = min(1,1100/monitor['width'],650/monitor['height'])
    canvas = tk.Canvas(root,width=round(monitor['width']*scale),
                       height=round(monitor['height']*scale),highlightthickness=0,bg='#181818')
    canvas.pack()
    footer = tk.Frame(root)
    footer.pack(fill='x')
    lens = tk.Canvas(footer,width=160,height=160,bg='black',highlightthickness=0)
    lens.pack(side='left')
    controls = tk.Frame(footer)
    controls.pack(side='left',fill='both',expand=True,padx=12)
    status = tk.StringVar(value='Preparing a clean game screenshot...')
    tk.Label(controls,textvariable=status,justify='left',wraplength=800).pack(anchor='w')
    radius = tk.IntVar(value=int(cfg.get('player_radius',6)))
    row = tk.Frame(controls)
    row.pack(anchor='w')
    tk.Label(row,text='Estimated hitbox radius (pixels):').pack(side='left')
    tk.Spinbox(row,from_=1,to=30,textvariable=radius,width=5).pack(side='left')
    tk.Label(controls,text='Radius is an estimate, not a hitbox extracted from game data.\n'
             'R: retake in 5 sec | Backspace: reset | Enter: save | Esc: cancel',
             justify='left').pack(anchor='w')
    state = dict(image=None,preview=None,lens=None,start=None,arena=None,player=None,hover=None,busy=False)
    def preview_radius():
        try:
            return max(1,min(30,radius.get()))
        except (ValueError,tk.TclError):
            return int(cfg.get('player_radius',6))
    def draw():
        canvas.delete('selection')
        if state['arena']:
            canvas.create_rectangle(*(v*scale for v in state['arena']),outline='yellow',width=2,tags='selection')
        if state['player']:
            x,y = state['player']
            r = preview_radius()
            canvas.create_oval((x-r)*scale,(y-r)*scale,(x+r)*scale,(y+r)*scale,
                               outline='#00ff88',width=2,tags='selection')
        point = state['player'] or state['hover']
        if point and state['image']:
            x,y = point
            crop = state['image'].crop((x-20,y-20,x+20,y+20)).resize((160,160),Image.Resampling.NEAREST)
            state['lens'] = ImageTk.PhotoImage(crop)
            lens.delete('all')
            lens.create_image(0,0,anchor='nw',image=state['lens'])
            lens.create_line(0,80,160,80,fill='yellow')
            lens.create_line(80,0,80,160,fill='yellow')
            r = preview_radius()*4
            lens.create_oval(80-r,80-r,80+r,80+r,outline='#00ff88',width=2)
    def reset(event=None):
        state.update(start=None,arena=None,player=None)
        status.set('1. Drag around the playable area, excluding HUD and chat.')
        draw()
    def capture_now():
        # Root stays withdrawn until after grab, preventing recursive screenshots.
        if sys.platform=='win32' and not focus_game():
            state['busy']=False
            root.deiconify()
            status.set('Could not focus RotMGExalt. Open the game, then press R to retry.')
            return
        # Wait for the game/compositor to paint after foreground activation.
        root.after(400,grab)
    def grab():
        try:
            with mss.mss() as capture:
                shot = capture.grab(monitor)
                image = Image.frombytes('RGB',shot.size,shot.rgb)
            state['image'] = image
            state['preview'] = ImageTk.PhotoImage(image.resize((round(image.width*scale),round(image.height*scale))))
            canvas.delete('all')
            canvas.create_image(0,0,anchor='nw',image=state['preview'])
            reset()
        except Exception as exc:
            status.set('Capture failed: '+str(exc))
        finally:
            state['busy']=False
            root.deiconify()
            root.lift()
            root.title('Select arena, then player center; magnifier below; Enter to save')
    def retake(event=None):
        if state['busy']:
            return
        state['busy']=True
        root.withdraw()
        focus_game()
        root.after(5000,capture_now)
    def point(e):
        return (round(e.x/scale),round(e.y/scale))
    def down(e):
        if not state['image'] or state['busy']:
            return
        x,y = point(e)
        if state['arena']:
            l,t,r,b = state['arena']
            if l<x<r and t<y<b:
                state['player']=(x,y)
                status.set(f'Center: {x}, {y}. Check the 4x magnifier. Click again to adjust; Enter saves.')
                draw()
        else:
            state['start']=(x,y)
    def move(e):
        state['hover']=point(e)
        if state['start'] and not state['arena']:
            canvas.delete('drag')
            canvas.create_rectangle(state['start'][0]*scale,state['start'][1]*scale,
                                    e.x,e.y,outline='yellow',width=2,tags='drag')
        draw()
    def up(e):
        if state['start'] and not state['arena']:
            x,y = point(e)
            sx,sy = state['start']
            l,r = sorted([sx,x]); t,b = sorted([sy,y])
            l,t,r,b = max(0,l),max(0,t),min(monitor['width'],r),min(monitor['height'],b)
            if r-l>=100 and b-t>=100:
                state['arena']=[l,t,r,b]
                status.set('2. Click the player center. Magnifier is below. Enter saves after selection.')
            state['start']=None
            canvas.delete('drag')
            draw()
    def save(event=None):
        if not state['arena'] or not state['player']:
            return
        try:
            r = radius.get()
            if not 1<=r<=30:
                raise ValueError('Radius must be 1–30 pixels')
            cfg.update(frame_size=[monitor['width'],monitor['height']],arena=state['arena'],
                       player=list(state['player']),player_radius=r,ignore_rectangles=[],monitor=monitor)
            path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        except Exception as exc:
            status.set(str(exc))
            return
        print(f'Calibration saved: {path}',flush=True)
        root.destroy()
    tk.Button(controls,text='Retake (5 sec)',command=retake).pack(side='left')
    tk.Button(controls,text='Reset selection',command=reset).pack(side='left')
    tk.Button(controls,text='Save',command=save).pack(side='left')
    canvas.bind('<ButtonPress-1>',down)
    canvas.bind('<Motion>',move)
    canvas.bind('<B1-Motion>',move)
    canvas.bind('<ButtonRelease-1>',up)
    root.bind('<Escape>',lambda e:root.destroy())
    root.bind('<Return>',save)
    root.bind('<BackSpace>',reset)
    root.bind('<r>',retake)
    root.bind('<R>',retake)
    root.after(100,retake)
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
