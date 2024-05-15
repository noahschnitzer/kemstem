import numpy as np
### normalization

#min=0,max=1, no type conversions
def normalize(data):
    data = data - np.min(data)
    data = data / np.max(data)
    return data
def normalize_sum(data):
    return data/data.sum()
def normalize_max(data):
    return data/data.max()
