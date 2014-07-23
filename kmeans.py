import matplotlib.pylab as plt
from numpy import *
from random import *
import distance


def kmeans(X, k, dist=distance.euclid):
    m, n = shape(X)
    centroids = mat([[randrange(min(X[:, i]), max(X[:, i])) for i in xrange(n)]
                     for x in xrange(k)])

    dist_label = zeros((m, 2))
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i, x in enumerate(X):
            d, l = min([(dist(x, c) ** 2, j) for j, c in enumerate(centroids)])
            cluster_changed = cluster_changed or l != dist_label[i, 1]
            dist_label[i] = d, l
        for i in xrange(k):
            sub_X = X[nonzero(dist_label[:, 1] == i)[0]]
            if len(sub_X) != 0:
                centroids[i, :] = mean(sub_X, axis=0)
    return dist_label, centroids


def bi_kmeans(X, k, dist=distance.euclid):
    m, n = shape(X)
    centroids = [[mean(X[:, i]) for i in xrange(n)]]
    dist_label = zeros((m, 2))
    for i, x in enumerate(X):
        dist_label[i] = dist(x, centroids[0]) ** 2, 0

    best_label = -1
    best_sse = inf
    best_sub_centroids = None
    while len(centroids) < k:
        for i, c in enumerate(centroids):
            sub_X = X[nonzero(dist_label[:, 1] == i)[0]]

            sub_dist_label, sub_centroids = kmeans(sub_X, 2)

            sse = sum(dist_label[nonzero(dist_label[:, 1] != i)[0]]) + sum(
                sub_dist_label)

            # find the best cluster
            if best_sse > sse:
                best_sse = sse
                best_label = i
                best_dist_label = sub_dist_label
                best_sub_centroids = sub_centroids

        # update centroid by the best cluster
        centroids[best_label] = best_sub_centroids[0]
        centroids.append(best_sub_centroids[1])

        # update the label of X
        for i, j in zip(nonzero(dist_label[:, 1] == best_label)[0], xrange(len(best_dist_label))):
            if best_dist_label[j][1] == 1:
                dist_label[i] = best_dist_label[j][0], len(centroids) - 1
            else:
                dist_label[i, 0] = best_dist_label[j][0]
    
    return dist_label, centroids
