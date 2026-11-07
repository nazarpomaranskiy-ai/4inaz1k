"""Foreground-only Windows movement with a short command lease and watchdog."""
import ctypes
from ctypes import wintypes
import math
import os
import sys
import threading
import time

SCANCODES = {'W':0x11,'A':0x1e,'S':0x1f,'D':0x20}


def choose_movement(observation, now, danger_clearance=8, max_age=.10):
    """Move only when staying is threatened and another action improves clearance."""
    if not observation or 'error' in observation or observation.get('uncertain'):
        return ''
    if not 0 <= now-observation['stamp'] <= max_age:
        return ''
    best = observation.get('best')
    if not best or not observation.get('tracks'):
        return ''
    stationary = next((r for r in observation.get('scores',[]) if r[0]=='STAY'),None)
    if stationary is None or not math.isfinite(stationary[1]):
        return ''
    name, gap, _, _ = best
    if stationary[1] >= danger_clearance or gap <= stationary[1]+2:
        return ''
    if name == 'STAY' or name not in ('W','A','S','D','WA','WD','SA','SD'):
        return ''
    return name


def apply_observation(movement,observation,now,paused=False,max_age=.1):
    """Temporary missing/uncertain observations release keys without losing F6 mode."""
    if paused:
        movement.disarm();return 'paused'
    if observation and 'error' in observation:
        movement.disarm();return 'capture error: '+str(observation['error'])
    if not observation:
        movement.update('');return 'waiting for first game frame'
    if observation.get('unfocused'):
        movement.update('');return 'waiting for focused game frame'
    age=now-observation['stamp']
    if age<0 or age>max_age:
        movement.update('');return f'frame age {age*1000:.0f} ms (limit {max_age*1000:.0f})'
    if observation.get('uncertain'):
        movement.update('');return observation.get('blocked_reason','vision capacity / camera uncertainty')
    movement.update(choose_movement(observation,now,max_age=max_age))
    return ''


class MovementState:
    """Pure lease state, separate from Windows for deterministic tests."""
    def __init__(self):
        self.armed = False
        self.direction = ''
        self.deadline = 0

    def desired(self, now, foreground, abort):
        if abort or not foreground:
            self.armed = False
            self.direction = ''
        if not self.armed or now > self.deadline:
            return set()
        return set(self.direction)


class WindowsMovement:
    def __init__(self,speed=180):
        if sys.platform != 'win32':
            raise RuntimeError('Windows input is only available on Windows')
        self.user32 = ctypes.WinDLL('user32',use_last_error=True)
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowTextW.argtypes = [wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        ulong_ptr = ctypes.c_size_t
        class Keyboard(ctypes.Structure):
            _fields_ = [('vk',wintypes.WORD),('scan',wintypes.WORD),('flags',wintypes.DWORD),
                        ('time',wintypes.DWORD),('extra',ulong_ptr)]
        class Mouse(ctypes.Structure):
            _fields_ = [('dx',wintypes.LONG),('dy',wintypes.LONG),('data',wintypes.DWORD),
                        ('flags',wintypes.DWORD),('time',wintypes.DWORD),('extra',ulong_ptr)]
        class Hardware(ctypes.Structure):
            _fields_ = [('msg',wintypes.DWORD),('low',wintypes.WORD),('high',wintypes.WORD)]
        class Payload(ctypes.Union):
            _fields_ = [('keyboard',Keyboard),('mouse',Mouse),('hardware',Hardware)]
        class Input(ctypes.Structure):
            _fields_ = [('type',wintypes.DWORD),('payload',Payload)]
        self.Input = Input
        self.user32.SendInput.argtypes = [wintypes.UINT,ctypes.POINTER(Input),ctypes.c_int]
        self.user32.SendInput.restype = wintypes.UINT
        self.state = MovementState()
        self.lock = threading.RLock()
        self.held = set()
        self.speed = speed
        self.offset = [0.0,0.0]
        self.last_motion = time.perf_counter()
        self.error = ''
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch,daemon=True)
        self.thread.start()

    def foreground(self):
        hwnd = self.user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        # GetWindowText on our own Tk window can send a synchronous message;
        # never make that call from a watchdog while the UI waits on its lock.
        if pid.value == os.getpid():
            return False
        buf = ctypes.create_unicode_buffer(512)
        self.user32.GetWindowTextW(hwnd,buf,512)
        return buf.value.replace(' ','').lower() == 'rotmgexalt'

    def _send(self,key,up):
        packet = self.Input()
        packet.type = 1
        packet.payload.keyboard.scan = SCANCODES[key]
        packet.payload.keyboard.flags = 0x8 | (0x2 if up else 0)
        if self.user32.SendInput(1,ctypes.byref(packet),ctypes.sizeof(packet)) != 1:
            raise RuntimeError('Windows rejected movement input (permissions or desktop changed)')

    def position(self):
        # Integrate commanded displacement for a camera that follows the player.
        with self.lock:
            now = time.perf_counter()
            vx,vy = self.velocity(self.speed)
            dt = now-self.last_motion
            self.offset[0] += vx*dt
            self.offset[1] += vy*dt
            self.last_motion = now
            return tuple(self.offset)

    def _apply(self,desired):
        self.position()
        for key in sorted(self.held-desired):
            self._send(key,True)
            self.held.remove(key)
        for key in sorted(desired-self.held):
            self._send(key,False)
            self.held.add(key)

    def _release(self):
        self.position()
        # Try every held key even if one release fails.
        for key in list(self.held):
            try:
                self._send(key,True)
                self.held.remove(key)
            except Exception as exc:
                self.error = str(exc)

    def arm(self):
        with self.lock:
            if self.error or not self.foreground():
                return False
            self.state.armed = True
            self.state.direction = ''
            self.state.deadline = 0
            return True

    def disarm(self):
        with self.lock:
            self.state.armed = False
            self.state.direction = ''
            self._release()

    def update(self,direction):
        with self.lock:
            self.state.direction = direction
            self.state.deadline = time.perf_counter()+.12

    def armed(self):
        with self.lock:
            return self.state.armed

    def velocity(self,speed):
        with self.lock:
            dx = int('D' in self.held)-int('A' in self.held)
            dy = int('S' in self.held)-int('W' in self.held)
            norm = math.hypot(dx,dy) or 1
            return speed*dx/norm,speed*dy/norm

    def _watch(self):
        try:
            while not self.stop.is_set():
                abort = any(self.user32.GetAsyncKeyState(key)&0x8000 for key in (0x77,0x78))
                foreground = self.foreground()
                with self.lock:
                    desired = self.state.desired(time.perf_counter(),foreground,abort)
                    self._apply(desired)
                self.stop.wait(.015)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.disarm()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.disarm()
