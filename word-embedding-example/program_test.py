import unittest
from program import TextML


class TextMLTest(unittest.TestCase):
    def setUp(self):
        self.textml = TextML()

    def test_train(self):
        self.textml.train()


if __name__ == '__main__':
    unittest.main()
