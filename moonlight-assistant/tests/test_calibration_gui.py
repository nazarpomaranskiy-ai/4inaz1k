"""Opt-in GUI regression: RUN_GUI_TESTS=1 on Windows or with a display."""
import json
import os
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch
import overlay

@unittest.skipUnless(os.environ.get('RUN_GUI_TESTS')=='1','GUI test requires an interactive display')
class CalibrationGuiTest(unittest.TestCase):
    def test_capture_hidden_then_confirm_selection(self):
        root = tk.Tk()
        original_after = root.after
        def fast_after(ms,callback=None,*args):
            return original_after({5000:50,400:10}.get(ms,ms),callback,*args)
        root.after = fast_after
        captures = []
        failures = []
        class Shot:
            size=(640,480)
            rgb=bytes(640*480*3)
        class Capture:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def grab(self,monitor):
                captures.append(root.state())
                return Shot()
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        def select():
            try:
                canvas=next(w for w in root.winfo_children() if isinstance(w,tk.Canvas))
                canvas.event_generate('<ButtonPress-1>',x=20,y=30)
                canvas.event_generate('<B1-Motion>',x=400,y=350)
                canvas.event_generate('<ButtonRelease-1>',x=400,y=350)
                canvas.event_generate('<ButtonPress-1>',x=200,y=220)
                next(w for w in descendants(root) if isinstance(w,tk.Button) and w.cget('text')=='Save').invoke()
            except Exception as exc:
                failures.append(str(exc))
                root.destroy()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'live.json'
            original_after(350,select)
            original_after(1500,lambda:root.destroy())
            with patch.object(overlay.tk,'Tk',return_value=root),patch.object(overlay.mss,'mss',Capture),patch.object(overlay,'focus_game',return_value=True):
                overlay.calibrate(dict(left=0,top=0,width=640,height=480),path)
            self.assertEqual(failures,[])
            self.assertEqual(captures,['withdrawn'])
            cfg=json.loads(path.read_text())
            self.assertEqual(cfg['arena'],[20,30,400,350])
            self.assertEqual(cfg['player'],[200,220])
