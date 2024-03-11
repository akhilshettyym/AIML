import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
data = pd.read_csv('/data/exe.csv')
f1 = data['v1'].values
f2 = data['v2'].values
X = np.array(list(zip()))
print('x:',X)
print('Graph for the whole dataset ')
plt.scatter(f1,f2,c = 'black')
plt.show()
kmeans = KMeans(2)
labels = kmeans.fit(X).predict(X)
print('Labels for kmeans :',labels)
print('Graph using Kmeans algorithm')
plt.scatter(f1,f2,c =labels)
centroids = kmeans.cluster_centers_
print('Centroids :',centroids)
plt.scatter(centroids[:,0],centroids[:,1],marker = '*',c='red')
plt.show()
gmm = GaussianMixture(2)
labels = gmm.fit(X).predict(X)
print('Labels for GMM :',labels)
print('Graph using EM algorithm')
plt.scatter(f1,f2,c=labels)
plt.show()