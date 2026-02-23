import random
import umap
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from scipy.spatial import procrustes

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

def signal_strength(base, ap):
    dist = ((ap.x - base.x)**2 + (ap.y - base.y)**2)**0.5
    #signal = 20 - dist
    #signal = 1 / (dist + 1e-6) # * ap.signal
    signal = -40 - (10 * 2 * np.log10(max(dist, 0.1))) * ap.signal # linear -> rssi + strength
    signal = 10 ** ((-40 - signal) / 20) # rssi + strength -> linear + strength
    return signal

for i in range(10):
    print("starting data build")
    
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
    
    x_fac = max([abs(base.x) for base in bases])
    y_fac = max([abs(base.y) for base in bases])
    print(x_fac)
    print(y_fac)
    
    data = {str(k.id): [] for k in aps}
    
    for base in bases:
        for ap in aps:
            dist = signal_strength(base, ap)
            data[str(ap.id)].append( dist)
    
    reducer = umap.UMAP(n_neighbors=7, n_epochs=50, learning_rate=0.25, min_dist=0.0, init='pca', spread=0.3, repulsion_strength=0.025, negative_sample_rate=5,  metric='euclidean', verbose=True,angular_rp_forest=True,local_connectivity=2)
    print("building dataframe")
    data = pd.DataFrame(data=data)
    print("fitting data")
    #data_fit = StandardScaler().fit_transform(data)
    print("making embedding")
    start = time.perf_counter()
    embedding = reducer.fit_transform(data)
    print(time.perf_counter() - start, "sec")
    print("scaling embedding")
    #embedding[:, 0] -= np.mean(embedding[:, 0])
    #embedding[:, 1] -= np.mean(embedding[:, 1])
    #embedding *= (10/np.max(np.abs(embedding), axis=0))
    print(embedding.shape)
    gt = np.array([[ap.x, ap.y] for ap in bases])
    gt_ap = np.array([[ap.x, ap.y] for ap in aps])
    #gt[:, 0] -= np.mean(gt[:, 0])
    #gt[:, 1] -= np.mean(gt[:, 1])
    #gt *= (10/np.max(np.abs(gt), axis=0))
    mtx1, mtx2, disparity = procrustes(gt, embedding)
    mtx1[:, 0] *= x_fac / max(mtx1[:, 0])
    mtx1[:, 1] *= y_fac / max(mtx1[:, 1])
    mtx2[:, 0] *= x_fac / max(mtx2[:, 0])
    mtx2[:, 1] *= y_fac / max(mtx2[:, 1])
    plt.scatter(mtx1[:, 0], mtx1[:, 1], label="gt", marker="x")
    plt.scatter(gt_ap[:, 0], gt_ap[:, 1], label="gt_ap", marker="x", c="green")
    plt.scatter(mtx2[:, 0], mtx2[:, 1], label="umap")
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('meowing', fontsize=24);
    plt.show()
