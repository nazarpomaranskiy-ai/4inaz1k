"""Capture the game window itself, excluding unrelated windows and overlay."""
import threading
import time


class WindowsGameCapture:
    def __init__(self, monitor, arena):
        self.monitor = monitor
        self.arena = arena
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.consumed = 0
        self.error = None
        self.control = None

    def __enter__(self):
        from windows_capture import WindowsCapture
        self.capture = WindowsCapture(cursor_capture=False,draw_border=False,
                                      window_name='RotMGExalt')
        @self.capture.event
        def on_frame_arrived(frame,control):
            try:
                if (frame.width,frame.height) != (self.monitor['width'],self.monitor['height']):
                    raise RuntimeError('Use borderless RotMG covering the calibrated monitor, then recalibrate')
                l,t,r,b = self.arena
                image = frame.frame_buffer[t:b,l:r,:3].copy()
                with self.lock:
                    self.sequence += 1
                    self.latest = (image,time.perf_counter())
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                control.stop()
        @self.capture.event
        def on_closed():
            with self.lock:
                self.error = 'Game capture closed; restart overlay after reopening the game'
        self.control = self.capture.start_free_threaded()
        return self

    def read(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.latest is None or self.consumed == self.sequence:
                return None
            self.consumed = self.sequence
            return self.latest

    def __exit__(self,*args):
        if self.control and not self.control.is_finished():
            self.control.stop()
