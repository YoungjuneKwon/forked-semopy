import unittest
from enum import Enum
import parser

class TestOperations(Enum):
    OP1 = '~'
    OP2 = '+!'

class TestParser(unittest.TestCase):
    def setUp(self):
        self.ops = TestOperations
        self.parser = parser.Parser(self.ops)

    def test_parser(self):
        command = 'a ~ b + c\nc +! 2*a + 3*b'
        cmd = self.parser.parse(command)
        self.assertTrue('b' in cmd['a'][self.ops.OP1])
        self.assertEqual(len(cmd['a'][self.ops.OP1]['b']), 0)
        self.assertEqual(len(cmd['a'][self.ops.OP1]['c']), 0)
        self.assertEqual(len(cmd['c'][self.ops.OP1]['a']), 1)
        self.assertEqual(len(cmd['c'][self.ops.OP1]['b']), 1)
        self.assertEqual(cmd['c'][self.ops.OP2]['a'][0], '2')
        self.assertEqual(cmd['c'][self.ops.OP2]['b'][0], '3')
        
        