import unittest
import numpy as np
import cv2
from tracking import AdaptiveTracker,CameraMotion
from planner import Shot,recommend

class AdaptiveTests(unittest.TestCase):
    def test_acceleration_is_learned_from_history(self):
        tracker=AdaptiveTracker()
        for i in range(24):
            t=i/30
            shots=tracker.update([(100+30*t+100*t*t,80,3)],1/30)
        self.assertEqual(len(shots),1)
        self.assertEqual(tracker.tracks[0].model,2)
        self.assertAlmostEqual(shots[0].ax,200,places=3)
        self.assertAlmostEqual(shots[0].vx,30+200*23/30,places=3)

    def test_linear_motion_does_not_need_acceleration(self):
        tracker=AdaptiveTracker()
        for i in range(20):shots=tracker.update([(20+i*3,30,3)],1/30)
        self.assertAlmostEqual(shots[0].vx,90,places=3)
        self.assertEqual(tracker.tracks[0].model,1)

    def test_short_occlusion_is_retained_with_increased_uncertainty(self):
        tracker=AdaptiveTracker()
        for i in range(6):shots=tracker.update([(20+i*3,30,3)],1/30)
        before=shots[0].uncertainty
        shots=tracker.update([],1/30)
        self.assertEqual(len(shots),1);self.assertGreater(shots[0].uncertainty,before)
        for _ in range(5):shots=tracker.update([],1/30)
        self.assertEqual(shots,[])

    def test_camera_translation_removed(self):
        rng=np.random.default_rng(42)
        frame=rng.integers(0,256,(200,300,3),dtype=np.uint8)
        moved=cv2.warpAffine(frame,np.float32([[1,0,6],[0,1,-4]]),(300,200))
        camera=CameraMotion((150,100));camera.update(frame);offset=camera.update(moved)
        self.assertGreater(camera.confidence,.6)
        np.testing.assert_allclose(offset,[-6,4],atol=.5)

    def test_accelerating_shot_threatens_stationary_player(self):
        _,scores=recommend((100,100),[Shot(40,100,0,0,3,ax=800)],(0,0,200,200),speed=100,horizon=.4)
        stay=next(s for s in scores if s[0]=='STAY')
        self.assertLess(stay[1],0)
