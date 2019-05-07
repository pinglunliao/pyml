# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:20:53 2018

@author: YunAIUser
"""
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

n_neighbors = 30
lle = LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')

X_r3 = lle.fit_transform(X)

# Percentage of variance explained for each components
# Percentage of variance explained for each components
print('Explained variance ratio for PCA (first two components): %s'
      % str(pca.explained_variance_ratio_))

print('Explained variance ratio for LDA (first two components): %s'
      % str(lda.explained_variance_ratio_))

print('Reconstruction error for LLE (first two components): %s'
      % str(lle.reconstruction_error_))

colors = ['navy', 'turquoise', 'darkorange']
markers = ['s', 'X', 'P']

def drawChart(data, title):    
    plt.figure()

    for color, i, target_name, marker in zip(colors, [0, 1, 2], target_names, markers):
        plt.scatter(data[y == i, 0], data[y == i, 1], color=color, alpha=.8,
                label=target_name, marker=marker)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title( title + ' of IRIS dataset')

drawChart(X_r, "PCA")
drawChart(X_r2, "LDA")
drawChart(X_r3, "LLE")

plt.show()