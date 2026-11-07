import unittest
from planner import Shot, recommend

class PlannerTests(unittest.TestCase):
    def test_no_observations_is_unknown(self):
        self.assertEqual(recommend((100,100), [], (0,0,200,200)), (None, []))

    def test_incoming_shot_requires_dodge(self):
        best, scores = recommend((100,100), [Shot(50,100,200,0,3)],
                                 (0,0,200,200), speed=100, horizon=.4)
        stay = next(s for s in scores if s[0]=='STAY')
        self.assertLess(stay[1],0)
        self.assertGreater(best[1],0)
        self.assertNotEqual(best[0],'STAY')

    def test_no_candidate_exits_arena(self):
        _, scores = recommend((8,100), [Shot(80,100,-100,0,3)],
                              (0,0,200,200), speed=100, horizon=.4)
        self.assertFalse({'A','WA','SA'} & {s[0] for s in scores})

    def test_diagonal_speed_is_normalized(self):
        import math
        _, scores = recommend((100,100), [Shot(180,180,0,0,2)],
                              (0,0,200,200), speed=100)
        for name, gap, vx, vy in scores:
            if name != 'STAY':
                self.assertAlmostEqual(math.hypot(vx,vy),100)

    def test_current_overlap_is_unsafe_for_every_action(self):
        best, _ = recommend((100,100), [Shot(100,100,100,0,5)], (0,0,200,200))
        self.assertLess(best[1],0)

if __name__ == '__main__':
    unittest.main()
