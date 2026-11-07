"""Explicit opt-in: sends W only to this test's own empty Tk window."""
import os
import sys
import tkinter as tk
import unittest
from autododge import WindowsMovement

@unittest.skipUnless(sys.platform=='win32' and os.environ.get('RUN_INPUT_TESTS')=='1',
                     'Requires Windows and explicit input-test opt-in')
class WindowsInputTest(unittest.TestCase):
    def test_scan_code_press_and_release_in_own_window(self):
        root=tk.Tk()
        title='Moonlight input self-test (empty window)'
        root.title(title)
        root.geometry('420x120+50+50')
        root.attributes('-topmost',True)
        tk.Label(root,text='Testing W key in this window only.').pack()
        root.update()
        root.focus_force()
        movement=WindowsMovement()
        events=[]
        errors=[]
        sent=[]
        root.bind_all('<KeyPress>',lambda e:events.append(('down',e.keycode)))
        root.bind_all('<KeyRelease>',lambda e:events.append(('up',e.keycode)))
        def send():
            import ctypes
            from ctypes import wintypes
            user32=movement.user32
            buf=ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(user32.GetForegroundWindow(),buf,512)
            if buf.value != title:
                errors.append('Test window is not focused; no keys were sent')
                return
            try:
                movement._send('W',False)
                sent.append(True)
                root.after(60,lambda:movement._send('W',True))
            except Exception as exc:
                errors.append(str(exc))
        root.after(250,send)
        root.after(650,root.destroy)
        try:
            root.mainloop()
        finally:
            if sent:
                movement._send('W',True)
            movement.close()
        self.assertEqual(errors,[])
        self.assertIn(('down',87),events)
        self.assertIn(('up',87),events)
