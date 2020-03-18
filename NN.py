import numpy.random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


def knn(train_images, labels, testImage, k):
    distance = [0] * len(train_images)
    for i in range(len(distance)):
        distance[i] = numpy.linalg.norm(train_images[i] - testImage)
    # attach label to image_distance
    neighbors = list(zip(distance, labels))
    # sort by distance
    neighbors.sort(key=lambda tup: tup[0])
    label_cnt = [0] * 10
    # find most common label
    # use only k nearest neighbors
    for i in range(k):
        label_cnt[int(neighbors[i][1])] += 1
    return label_cnt.index(max(label_cnt))


# k = 1 works best
def calc_accuracy(k, n):
    err = 0
    for i in range(len(test)):
        if (int(test_labels[i]) != knn(train[:n], train_labels[:n], test[i], k)):
            err += 1
    accuracy = 1 - err / len(test)
    return accuracy


if __name__ == '__main__':
    # print("accuracy = " + str(calc_accuracy(10, 1000)))

    # plot accuracy as func of k
    # plt.plot([i for i in range(1, 101)], [calc_accuracy(k, 1000) for k in range(1, 101)])
    # plt.show()

    # plot accuracy as func of n
    plt.plot([i for i in range(100, 5001, 100)], [calc_accuracy(1, n) for n in range(100, 5001, 100)])
    plt.show()
