from collections import defaultdict
from itertools import islice


class ThreadsManager:
    '''A helper class used to keep track of different threads in graph. Needed
    to build cycles if neccessary.
    '''
    def __init__(self, dicts=list()):
        self.threads = list()
        for d in dicts:
            self.load_from_dict(d)

    def load_from_dict(self, d: dict, reverse=False):
        for a, to_nodes in d.items():
            for b in to_nodes:
                if not reverse:
                    self.connect_nodes(a, b)
                else:
                    self.connect_nodes(b, a)

    def add_node(self, node: str):
        for thread in self.find_threads(node):
            return
        self.threads.append([node])

    def connect_nodes(self, a: str, b: str):
        is_var_present = False
        for thread in islice(self.threads, len(self.threads)):
            if a in thread:
                is_var_present = True
                if thread[-1] == a:
                    thread.append(b)
                else:
                    i = thread.index(a)
                    lt = thread[:i + 1]
                    lt.append(b)
                    # When cycles are present duplicates are possible.
                    if lt not in self.threads:
                        self.threads.append(lt)
            elif thread[0] == b:
                is_var_present = True
                thread.insert(0, a)
        if not is_var_present:
            self.threads.append([a, b])

    def find_threads(self, node: str):
        for thread in self.threads:
            if node in thread:
                yield thread

    def translate_to_dict(self):
        d = defaultdict(set)
        for thread in self.threads:
            it = iter(thread)
            prev = next(it)
            for v in it:
                d[v].add(prev)
                prev = v
        return dict(d)

    def get_node_order(self, node: str):
        return max(thread.index(node) for thread in self.find_threads(node))

    def get_confluent_path(self, source_node: str):
        if source_node:
            threads = [thread[thread.index(source_node):]
                       for thread in self.find_threads(source_node)]
        else:
            threads = self.threads
        m = max(len(thread) for thread in threads)
        for i in range(m):
            yield [thr[i] if len(thr) > i else None for thr in threads]


def get_tuple_index(l, index, value):
    for pos, t in enumerate(l):
        if t[index] == value:
            return pos
