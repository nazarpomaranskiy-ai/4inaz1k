import unittest
from autododge import MovementState,choose_movement

class AutododgeTests(unittest.TestCase):
    def observation(self,gap=-4,bestgap=20):
        return dict(stamp=10,tracks=3,best=('W',bestgap,0,-180),
                    scores=[('STAY',gap,0,0),('W',bestgap,0,-180)])

    def test_moves_only_for_threat(self):
        self.assertEqual(choose_movement(self.observation(),10.03),'W')
        self.assertEqual(choose_movement(self.observation(gap=12),10.03),'')
        self.assertEqual(choose_movement(self.observation(gap=0,bestgap=1),10.03),'')

    def test_unknown_error_and_stale_release(self):
        for value in [None,{'error':'capture failed'},self.observation()|{'tracks':0}]:
            self.assertEqual(choose_movement(value,10.03),'')
        self.assertEqual(choose_movement(self.observation(),10.2),'')
        self.assertEqual(choose_movement(self.observation(),9.9),'')

    def test_lease_expiration_without_ui_update(self):
        state=MovementState(); state.armed=True; state.direction='WA'; state.deadline=10.12
        self.assertEqual(state.desired(10.1,True,False),{'W','A'})
        self.assertEqual(state.desired(10.13,True,False),set())

    def test_focus_loss_disarms_and_needs_explicit_rearm(self):
        state=MovementState(); state.armed=True; state.direction='D'; state.deadline=12
        self.assertEqual(state.desired(10,False,False),set())
        self.assertFalse(state.armed)
        self.assertEqual(state.desired(10.1,True,False),set())

    def test_emergency_key_disarms(self):
        state=MovementState(); state.armed=True; state.direction='S'; state.deadline=12
        self.assertEqual(state.desired(10,True,True),set())
        self.assertFalse(state.armed)
