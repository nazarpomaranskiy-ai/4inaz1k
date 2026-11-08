import unittest
from autododge import apply_observation

class FakeMovement:
    def __init__(self):self.enabled=True;self.direction='W'
    def update(self,direction):self.direction=direction
    def disarm(self):self.enabled=False;self.direction=''

class ActivationTests(unittest.TestCase):
    def test_stale_frame_releases_keys_without_cancelling_f6(self):
        movement=FakeMovement()
        reason=apply_observation(movement,dict(stamp=10),10.2)
        self.assertTrue(movement.enabled)
        self.assertEqual(movement.direction,'')
        self.assertIn('200 ms',reason)

    def test_missing_and_uncertain_frames_keep_mode_waiting(self):
        for obs in [None,dict(stamp=10,uncertain=True,blocked_reason='too many candidates')]:
            movement=FakeMovement();reason=apply_observation(movement,obs,10.01)
            self.assertTrue(movement.enabled);self.assertEqual(movement.direction,'');self.assertTrue(reason)

    def test_pause_and_capture_error_disarm(self):
        for obs,paused in [(dict(stamp=10),True),({'error':'failed'},False)]:
            movement=FakeMovement();apply_observation(movement,obs,10.01,paused)
            self.assertFalse(movement.enabled)

    def test_new_valid_frame_resumes_requested_mode(self):
        movement=FakeMovement();apply_observation(movement,None,10)
        obs=dict(stamp=10.02,tracks=3,best=('D',20,180,0),scores=[('STAY',-3,0,0),('D',20,180,0)])
        self.assertEqual(apply_observation(movement,obs,10.03),'')
        self.assertEqual(movement.direction,'D');self.assertTrue(movement.enabled)

    def test_recent_plan_from_128ms_capture_can_run(self):
        movement=FakeMovement()
        obs=dict(stamp=10,decision_stamp=10.11,tracks=3,best=('D',20,180,0),scores=[('STAY',-3,0,0),('D',20,180,0)])
        self.assertEqual(apply_observation(movement,obs,10.128),'')
        self.assertEqual(movement.direction,'D')

    def test_recent_plan_does_not_hide_old_capture(self):
        movement=FakeMovement()
        reason=apply_observation(movement,dict(stamp=10,decision_stamp=10.29),10.3)
        self.assertIn('frame age',reason);self.assertEqual(movement.direction,'')

    def test_old_plan_is_rejected_even_with_acceptable_capture_age(self):
        movement=FakeMovement()
        reason=apply_observation(movement,dict(stamp=10,decision_stamp=10.01),10.15)
        self.assertIn('plan age',reason);self.assertEqual(movement.direction,'')

    def test_pilot_strafe_duration_and_cooldown(self):
        from autododge import BurstLimiter
        limiter=BurstLimiter(.06,.1)
        self.assertEqual(limiter.command('D',10),'D')
        self.assertEqual(limiter.command('D',10.04),'D')
        self.assertEqual(limiter.command('D',10.07),'')
        self.assertEqual(limiter.command('D',10.12),'')
        self.assertEqual(limiter.command('D',10.18),'D')
