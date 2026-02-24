import random
import umap
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from scipy.spatial import procrustes
import threading
import random

class AP:
    def __init__(self, id):
        self.x = random.randint(-5000, 5000) / 100
        self.y = random.randint(-5000, 5000) / 100
        self.id = id
        self.signal = random.randint(10, 20) / 10

class Base:
    def __init__(self, x, y):
        #self.x = random.randint(-1000, 1000) / 100
        #self.y = random.randint(-1000, 1000) / 100
        self.x = x
        self.y = y

class ThreadHolder:
    def __init__(self):
        self.value = None

def signal_strength(base, ap):
    dist = ((ap.x - base.x)**2 + (ap.y - base.y)**2)**0.5
    #signal = 20 - dist
    #signal = 1 / (dist + 1e-6) # * ap.signal
    signal = -40 - (10 * 2 * np.log10(max(dist, 0.1))) * ap.signal # linear -> rssi + strength
    signal = 10 ** ((-40 - signal) / 20) # rssi + strength -> linear + strength
    return signal

print("starting data build")

all_data = [None]*3

def build_data(idx):
    aps = [AP(x) for x in range(50)]
    #bases = [Base() for x in range(10000)]
    s_x, s_y = 0, 0
    bases = []
    for b in range(2000):
        bases.append(Base(s_x, s_y))
        if not b % 200:        
            v_x = random.randint(-100, 100) / 1000
            v_y = random.randint(-100, 100) / 1000
        s_x += v_x
        s_y += v_y
    
    data = {str(k.id): [] for k in aps}
    for base in bases:
        for ap in aps:
            dist = signal_strength(base, ap)
            data[str(ap.id)].append( dist)
    data = pd.DataFrame(data=data)
    all_data[idx] = (data, bases, aps)

print("building test data")
data_threads = [threading.Thread(target=build_data, args=[x]) for x in range(3)]
[x.start() for x in data_threads]
[x.join() for x in data_threads]
print("data", all_data)

def loss_runner(holder, d_full, **vals):
    data, bases, aps = d_full
    reducer = umap.UMAP(**vals)
    embedding = reducer.fit_transform(data)
    gt = np.array([[ap.x, ap.y] for ap in bases])
    gt_ap = np.array([[ap.x, ap.y] for ap in aps])
    x_fac = max([abs(base.x) for base in bases])
    y_fac = max([abs(base.y) for base in bases])
    mtx1, mtx2, disparity = procrustes(gt, embedding)
    mtx1[:, 0] *= x_fac / max(mtx1[:, 0])
    mtx1[:, 1] *= y_fac / max(mtx1[:, 1])
    mtx2[:, 0] *= x_fac / max(mtx2[:, 0])
    mtx2[:, 1] *= y_fac / max(mtx2[:, 1])
    d1 = (mtx2[:, 0] - mtx1[:, 0])**2
    d2 = (mtx2[:, 0] - mtx1[:, 0])**2
    dist = np.sum((d2 + d1)**0.5)
    holder.value = dist
    

def get_loss(**vals) -> float:
    print("running loss")
    threads = []
    all_loss = [ThreadHolder() for x in range(len(all_data))]
    for idx, data_pack in enumerate(all_data):
        threads.append(threading.Thread(target=loss_runner, args=[all_loss[idx], data_pack], kwargs=vals))
    [x.start() for x in threads]
    [x.join() for x in threads]
    print([x.value for x in all_loss])
    return sum([x.value for x in all_loss])

class Done:
    def __init__(self):
        pass

class IntAllSearch:
    def __init__(self, low, high, val):
        self.low = low
        self.high = high
        self.val = val
    def search(self, vars):
        min_loss = None
        val = None
        for i in range(self.low, self.high+1):
            loss = get_loss(**vars, **{self.val: i})
            if min_loss == None or i < min_loss:
                min_loss = i
                val = i
        return val, min_loss

class FloatTriSearch:
    def __init__(self, low, high, val, f_depth, bound):
        self.low = low
        self.high = high
        self.val = val
        self.f_depth = f_depth
        self.bound = bound
    def expand_tri_search(self, low, high, m_loss, vars, depth):
        if depth > self.f_depth:
            return (low + (high - low) / 2, m_loss)
        l_loss = get_loss(**vars, **{self.val: low})
        h_loss = get_loss(**vars, **{self.val: high})
        min_loss = min(l_loss, m_loss, h_loss)
        delta = (high - low) / 4
        if min_loss == l_loss:
            l_delta = low - delta
            if self.bound and l_delta < self.low: l_delta = 0
            return self.expand_tri_search(l_delta, l_loss, low + delta, vars, depth+1)
        elif min_loss == h_loss:
            return self.expand_tri_search(high - delta, h_loss, high + delta, vars, depth+1)
        else:
            mid = (low + (high - low) / 2)
            return self.expand_tri_search(mid - delta, m_loss, mid + delta, vars, depth+1)
    def search(self, vars):
        mid = (self.low + (self.high - self.low) / 2)
        m_loss = get_loss(**vars, **{self.val: mid})
        return self.expand_tri_search(self.low, self.high, m_loss, vars, 0)

class StringSearch:
    def __init__(self, val, *strings):
        self.strings = strings
        self.val = val
    def search(self, vars):
        min_loss = None
        val = None
        for i in self.strings:
            loss = get_loss(**vars, **{self.val: i})
            if loss == None or i < min_loss:
                min_loss = loss
                val = i
        return val, min_loss
                
class BoolSearch:
    def __init__(self, val):
        self.val = val
    def search(self, vars):
        on_loss = get_loss(**vars, **{self.val: True})
        off_loss = get_loss(**vars, **{self.val: False})
        if on_loss < off_loss:
            return True, on_loss
        else:
            return False, off_loss

set_vars = {
    "n_epochs": 50,
    "init": "pca",
    "verbose": True,
}

opt_default_vars = {
    "n_neighbors": 7,
    "learning_rate": 0.25,
    "min_dist": 0.0,
    "spread": 0.3,
    "repulsion_strength": 0.025,
    "negative_sample_rate": 5,
    "metric": "euclidean",
    "angular_rp_forest": True,
    "local_connectivity": 2,
}

opt_vars = {
    "n_neighbors": IntAllSearch(1, 20, "n_neighbors"),
    "learning_rate": FloatTriSearch(0.0, 1.0, "learning_rate", 5, True),
    "min_dist": FloatTriSearch(0.0, 0.2, "min_dist", 5, True),
    "spread": FloatTriSearch(0.2, 1.0, "spread", 5, True),
    "repulsion_strength": FloatTriSearch(0.0, 1.0, "repulsion_strength", 5, True),
    "negative_sample_rate": IntAllSearch(1, 20, "negative_sample_rate"),
    "metric": StringSearch("metric", "euclidean", "cosine"),
    "angular_rp_forest": BoolSearch("angular_rp_forest"),
    "local_connectivity": IntAllSearch(1, 20, "local_connectivity"),
}
    
var_order = list(opt_vars.keys())
random.shuffle(var_order)
print("opt order", var_order)
for opt in var_order:
    vars = set_vars.copy()
    for x, y in opt_default_vars.items():
        if x not in vars.keys() and x != opt:
            vars[x] = y
    print("Running with:", vars)
    opted, loss = opt_vars[opt].search(vars)
    print("loss for", opt, loss)
    set_vars[opt] = opted

print(set_vars)
