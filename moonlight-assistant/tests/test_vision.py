import unittest
import numpy as np
import cv2
from analyze import Tracker, detect
from overlay import local_config

class VisionTests(unittest.TestCase):
    def test_real_frame_detection_and_velocity(self):
        cfg = dict(bullet_hsv=[[[0,140,160],[10,255,255]]],arena=[0,0,200,200],
                   min_area=5,max_area=180)
        tracker = Tracker(35,1400)
        for i in range(4):
            frame = np.zeros((200,200,3),np.uint8)
            cv2.circle(frame,(50+i*5,80),4,(0,0,255),-1)
            found,_ = detect(frame,cfg)
            self.assertEqual(len(found),1)
            shots = tracker.update(found,1/30)
        self.assertEqual(len(shots),1)
        self.assertAlmostEqual(shots[0].vx,150,places=3)
        self.assertAlmostEqual(shots[0].vy,0)
        self.assertEqual(tracker.update([],1/30),[])

    def test_region_coordinates_and_ignore_clipping(self):
        cfg = dict(arena=[100,50,500,350],player=[200,150],
                   ignore_rectangles=[[0,0,120,80],[600,600,700,700]])
        c = local_config(cfg)
        self.assertEqual(c['arena'],[0,0,400,300])
        self.assertEqual(c['player'],[100,100])
        self.assertEqual(c['ignore_rectangles'],[[0,0,20,30]])
        self.assertEqual(cfg['arena'],[100,50,500,350])

if __name__ == '__main__':
    unittest.main()
