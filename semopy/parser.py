'''This module contains a Parser class that is used to parse a semopy syntax.'''
import re
from collections import defaultdict
from enum import Enum


class ExampleOperations(Enum):
    REGRESSION = '~'
    MEASUREMENT = '=~'


class Parser:
    '''Parser is responsible for parsing a semopy syntax.
    
    Keyword arguments:
        
        operations -- An Enum containing operation names and their respective
                      symbols.
    '''
    s_pattern = r'\b(\S+(?:\s*\,\s*\S+)*)\s*({})\s*(\S+(?:\s*(?:\+|\*)\s*\S+)*)\b'
    
    def __init__(self, operations=ExampleOperations):
        '''
        Keyword arguments:
            
            operations -- An Enum containing operation names and their respective
                      symbols.
        '''
        self.m_opDict = {}
        self.operations = operations
        s = []
        for op in self.operations:
            self.m_opDict[op.value] = op
            beg_char = op.value[0].isalpha()
            end_char = op.value[-1].isalpha()
            val = ''
            if beg_char or end_char:
                if beg_char:
                    val += '(?<=\s)'
                val += op.value
                if end_char:
                    val += '(?=\s)'
            else:
                val = op.value
            s.append(val)
        self.m_pattern = self.s_pattern.format('|'.join(s))

    def parse(self, string: str):
        """"Parses a SEM model description provided in a appropirate syntax.
        
        Key arguments:
            
            string -- A model's description.
            
        Returns:
            
            A dictionary in the form of:
                lvalue -> TypeOfOperation -> rvalue -> ListOfExtraParameters
        """
        # Ignore comments.
        strings = [v.split('#')[0] for v in string.splitlines()]
        d = defaultdict(lambda: {op: defaultdict(lambda: []) for op in self.operations})
        for s in strings:
            r = re.search(self.m_pattern, s)
            if r:
                lvalues = [val.strip() for val in r.group(1).split(',')]
                rvalues = [val.strip().split('*')
                           for val in r.group(3).split('+')]
                op = self.m_opDict[r.group(2)]
                for lvalue in lvalues:
                    for rvalue in rvalues:
                        d[rvalue[-1]]
                        d[lvalue][op][rvalue[-1]] += rvalue[:-1]
        return d
