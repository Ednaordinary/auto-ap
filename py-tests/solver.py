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
        self.x = random.randint(-1000, 1000) / 10
        self.y = random.randint(-1000, 1000) / 10
        self.id = id
        self.signal = random.randint(10, 20) / 10

class Base:
    def __init__(self, x, y):
        #self.x = random.randint(-1000, 1000) / 100
        #self.y = random.randint(-1000, 1000) / 100
        self.x = x
        self.y = y
        self.measure = []
    def add_measure(self, m):
        self.measure.append(m)

def signal_strength(base, ap):
    dist = ((ap.x - base.x)**2 + (ap.y - base.y)**2)**0.5
    signal = 20 - dist
    #signal = 1 / (dist + 1e-6) # * ap.signal
    #signal = -40 - (10 * 2 * np.log10(max(dist, 0.1))) # linear -> rssi
    #signal = 10 ** ((-40 - signal) / 20) # rssi -> linear
    return signal

print("starting data build")

aps = [AP(x) for x in range(100)]
#bases = [Base() for x in range(10000)]
s_x, s_y = 0, 0
bases = []
for b in range(1000):
    bases.append(Base(s_x, s_y))
    if not b % 100:        
        v_x = random.randint(-100, 100) / 1000
        v_y = random.randint(-100, 100) / 1000
    s_x += v_x
    s_y += v_y

print(max([abs(base.x) for base in bases]))
print(max([abs(base.y) for base in bases]))

data = {str(k.id): [] for k in aps}

for base in bases:
    for ap in aps:
        dist = signal_strength(base, ap)
        data[str(ap.id)].append( dist)

reducer = umap.UMAP(n_neighbors=7, n_epochs=500, learning_rate=0.4, min_dist=0.0, init='pca', spread=0.25, repulsion_strength=0.05, target_metric='l2', metric='cosine', verbose=True,angular_rp_forest=True)
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
#gt[:, 0] -= np.mean(gt[:, 0])
#gt[:, 1] -= np.mean(gt[:, 1])
#gt *= (10/np.max(np.abs(gt), axis=0))
mtx1, mtx2, disparity = procrustes(gt, embedding)
plt.scatter(mtx1[:, 0], mtx1[:, 1], label="gt", marker="x")
plt.scatter(mtx2[:, 0], mtx2[:, 1], label="umap")
plt.gca().set_aspect('equal', 'datalim')
plt.title('meowing', fontsize=24);
plt.show()
