import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from observation import Budget, Analyzer
from observe_game import record_session
from tracking import CameraMotion

CFG={'monitor':{'width':128,'height':96},'arena':[0,0,128,96],'player':[64,48]}

class Capture:
    def __init__(self,*args):self.image=np.zeros((96,128,3),np.uint8)
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return self.image.copy(),time.perf_counter()

class Keys:
    def read(self):return {'game_foreground':True,'wasd':['W'],'stop':False}

class Detector:
    scene_supported=True
    boxes=[[30.,30.,40.,40.,.55]]
    def detect(self,frame):return [(35.,35.,5.)],None
    def close(self):pass

class ObservationTests(unittest.TestCase):
    def test_budget_existing_sessions_are_not_deleted_and_quota_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'existing';p.write_bytes(b'x'*100)
            b=Budget(tmp,max_mb=1,total_mb=100/1024**2,reserve_mb=0)
            self.assertEqual(b.reason(1,free=10000),'storage_limit')
            self.assertTrue(p.exists())
    def test_budget_checks_incoming_bytes_and_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Budget(tmp,max_mb=1,total_mb=2,reserve_mb=1)
            self.assertEqual(b.reason(100,free=1024**2+99),'low_disk_space')
            with patch('observation.shutil.disk_usage') as usage:
                usage.return_value.free=10*1024**2
                self.assertEqual(b.claim(1024**2),'')
                self.assertEqual(b.claim(1),'storage_limit')
    def test_recording_preserves_manual_controls_and_unreviewed_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=record_session(CFG,tmp,seconds=.3,fps=30,max_mb=1,total_mb=2,reserve_mb=0,
                               detector_factory=Detector,capture_factory=Capture,key_reader=Keys())
            report=json.loads((out/'report.json').read_text())
            rows=[json.loads(s) for s in (out/'frames.jsonl').read_text().splitlines()]
            analysis=[json.loads(s) for s in (out/'observations.jsonl').read_text().splitlines()]
            self.assertGreater(len(rows),1);self.assertTrue(analysis)
            self.assertEqual(rows[0]['wasd'],['W']);self.assertFalse(report['game_input'])
            self.assertFalse(analysis[0]['labels_reviewed']);self.assertFalse(report['automatic_training'])
            self.assertTrue((out/'review/queue.json').exists())
            self.assertLessEqual(report['budgeted_bytes'],1024**2)
    def test_detector_failure_does_not_lose_recording(self):
        def broken():raise RuntimeError('missing checkpoint')
        with tempfile.TemporaryDirectory() as tmp:
            out=record_session(CFG,tmp,seconds=.12,fps=30,max_mb=1,total_mb=2,reserve_mb=0,
                               detector_factory=broken,capture_factory=Capture)
            report=json.loads((out/'report.json').read_text())
            self.assertGreater(report['frames'],0)
            self.assertIn('missing checkpoint',report['analysis']['error'])
    def test_f9_saves_and_stops(self):
        class Stop:
            def read(self):return {'game_foreground':True,'wasd':[],'stop':True}
        with tempfile.TemporaryDirectory() as tmp:
            out=record_session(CFG,tmp,seconds=10,reserve_mb=0,capture_factory=Capture,key_reader=Stop())
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['stop_reason'],'F9');self.assertEqual(report['frames'],0)
    def test_camera_translation_is_reported_in_native_pixels(self):
        rng=np.random.default_rng(42)
        frame=rng.integers(0,256,(720,1555,3),dtype=np.uint8)
        shifted=cv2.warpAffine(frame,np.float32([[1,0,12],[0,1,-8]]),(1555,720))
        camera=CameraMotion([774,382]);camera.update(frame);camera.update(shifted)
        self.assertGreater(camera.confidence,.6)
        np.testing.assert_allclose(camera.shift,[12,-8],atol=1.5)

if __name__=='__main__':unittest.main()
