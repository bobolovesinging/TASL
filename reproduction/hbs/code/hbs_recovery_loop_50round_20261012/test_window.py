import unittest
from fifty_store import recent_rounds

class WindowTests(unittest.TestCase):
    def test_fifty_round_window_boundary(self):
        self.assertEqual(recent_rounds(50,5),set(range(46,51)))
    def test_window_larger_than_history(self):
        self.assertEqual(recent_rounds(3,5),{1,2,3})

if __name__=='__main__':unittest.main()
