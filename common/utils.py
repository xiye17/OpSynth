import pickle

def easy_pickle_read(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def easy_pickle_dump(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
