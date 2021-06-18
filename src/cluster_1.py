import pandas as pd
import numpy as np
import sklearn

from sklearn import cluster
from sklearn import datasets

import traceback


np.random.seed(0)
n_samples = 1500
random_state = 0
noise = 0.5

circle = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=noise, random_state=random_state)
moons = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
no_structures = np.random.rand(n_samples, 2), None
print(circle)
print(circle[0].shape)
print(moons)
print(moons[0].shape)
print(no_structures)
print(no_structures[0].shape)

try:
    model = cluster.KMeans(n_clusters=2, random_state=random_state)
    X, y = circle
    model.fit(circle)
    #lables = model.predict(X)
    #print(lables)
except:
    print(traceback.format_exc())