import pickle

class Persisted_dict(object):
    def __init__(self, filename, unsafe):
        self.filename = filename
        self.unsafe = unsafe
        self.cache = {}
        self.stored = {}
        self.len_cache = 0
        try:
            with open(filename, 'rb') as f:
                while True:
                    try:
                        (key, entry) = pickle.load(f)
                        self.stored[key] = entry
                    except:
                        break
            print ('Read a dictionary')
        except:
            print ('New dictionary created')
    
    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        else: return self.stored[key]
    
    def __contains__(self, key):
        return ((key in self.cache) or (key in self.stored))
    
    def _save_ (self):
        with open(self.filename, 'ab') as f:
            for key in self.cache:
                pickle.dump((key, self.cache[key]), f)
                self.stored[key] = self.cache[key]
        self.cache = {}
        self.len_cache = 0
                
                
    def __setitem__(self, key, value):
        self.cache[key] = value
        self.len_cache += 1
                
        if self.len_cache > self.unsafe:
            self._save_()
                
    def keys(self):
        self._save_()
        return list(self.stored.keys())
    
    def __iter__(self):
        self._save_()
        return self.stored.__iter__()
