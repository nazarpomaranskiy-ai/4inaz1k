import unittest
import cv2
import numpy as np
from realm_pilot import RealmPilot

class RealmPilotTests(unittest.TestCase):
    def image(self,x=80):
        frame=np.full((180,260,3),(25,80,25),np.uint8)
        cv2.rectangle(frame,(x,80),(x+9,85),(0,0,255),-1)
        cv2.rectangle(frame,(x+10,80),(x+14,85),(0,255,255),-1)
        # Pink and green sprites must never become this projectile class.
        cv2.circle(frame,(180,80),8,(220,80,255),-1)
        return frame
    def test_static_and_camera_translated_objects_rejected(self):
        engine=RealmPilot({'pilot_require_reference':False})
        self.assertEqual(engine.detect(self.image())[0],[])
        self.assertEqual(engine.detect(self.image(),camera_confidence=1)[0],[])
        moved=cv2.warpAffine(self.image(),np.float32([[1,0,6],[0,1,3]]),(260,180))
        self.assertEqual(engine.detect(moved,shift=(6,3),camera_confidence=1)[0],[])
    def test_moving_firebolt_detected_but_other_colors_rejected(self):
        engine=RealmPilot({'pilot_require_reference':False});engine.detect(self.image())
        found,_=engine.detect(self.image(100),camera_confidence=1)
        self.assertEqual(len(found),1)
        self.assertAlmostEqual(found[0][0],107,delta=2)
    def test_no_motion_claim_without_camera_evidence(self):
        engine=RealmPilot({'pilot_require_reference':False});engine.detect(self.image())
        self.assertEqual(engine.detect(self.image(100),camera_confidence=0)[0],[])

    def test_reference_rejects_different_moving_red_yellow_shape(self):
        from realm_pilot import appearance_descriptor
        frame=self.image()
        engine=RealmPilot({})
        prototype=appearance_descriptor(frame[78:88,78:98])
        engine.bank=np.array([prototype])
        other=np.full((180,260,3),(25,80,25),np.uint8)
        cv2.circle(other,(80,80),8,(0,0,255),-1)
        cv2.circle(other,(80,80),4,(0,255,255),-1)
        engine.detect(other)
        next_frame=np.full_like(other,(25,80,25))
        cv2.circle(next_frame,(105,80),8,(0,0,255),-1)
        cv2.circle(next_frame,(105,80),4,(0,255,255),-1)
        found,_=engine.detect(next_frame,camera_confidence=1)
        self.assertEqual(found,[])
