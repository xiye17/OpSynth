import pickle
import os

class SynthCache:
    def __init__(self, filename):
        self.cache = {}
        self.filename = filename
    
    def query(self, prob_id, ast_rep):
        key = '{}-{}'.format(prob_id, ast_rep)
        
        return self.cache.get(key, None)
    
    def write(self, prob_id, ast_rep, r):
        key = '{}-{}'.format(prob_id, ast_rep)
        
        self.cache[key] = r

    @classmethod
    def from_file(cls, filename):
        if os.path.isfile(filename):
            print('Load Cache')
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            print('New Cache')
            return cls(filename)
    
    def dump(self):
        other = SynthCache.from_file(self.filename)
        self.merge(other)
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)

    def merge(self, other):
        for k in other.cache:
            if k in self.cache:
                if self.cache[k] != other.cache[k]:
                    print('Inconsistentcy', k)
            else:
                self.cache[k] = other.cache[k]
