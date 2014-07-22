import matplotlib.pylab as plt
from numpy import *
from random import *
import distance


def k_means(X, k, dist=distance.euclid):
    m, n = shape(X)
    centroids = mat([[randrange(min(X[:, i]), max(X[:, i])) for i in xrange(n)]
                     for x in xrange(k)])
    # Label of X
    L = mat(zeros((m, 1)))
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False

        for idx, x in enumerate(X):
            l = mat(min([(dist(x, c), i) for i, c in enumerate(centroids)])[1])
            if l != L[idx, :]:
                cluster_changed = True
                L[idx, :] = l

        for i in xrange(k):
            centroids[i, :] = mean(X[nonzero(L[:, 0].A == i)[0]], axis=0)

    return L
