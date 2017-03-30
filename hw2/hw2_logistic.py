#!/usr/local/bin/python3
import sys
import math
import numpy

MAX_ITERATION = 5000

def read_labels(path):
    return numpy.genfromtxt(path, delimiter = ',', dtype = float, skip_header = 0)

def read_features(path):
    return numpy.genfromtxt(path, delimiter = ',', dtype = float, skip_header = 1)

def normalize(data, mean = None, sigma = None):
    if mean == None and sigma == None:
        mean = numpy.mean(data)
        sigma = numpy.std(data, ddof = 1)
    return numpy.divide(numpy.add(data, -1 * mean), sigma), mean, sigma

def sigmoid(values):
    return 1.0 / (1.0 + numpy.exp(-values))

def train(train_labels, features):
    e = 1e-31
    bias = 0.087
    model = {}
    learning_rate = 0.03
    i = bias_sum = best_accuracy = 0
    train_labels = train_labels.reshape(-1, 1)
    weights_sum = numpy.zeros((features.shape[1], 1))
    weights = numpy.array([1] * features.shape[1] ).reshape(features.shape[1], 1)
    while i < MAX_ITERATION:
        i = i + 1
        labels = sigmoid(numpy.add(bias, numpy.dot(features, weights)))
        differences = train_labels - labels
        b = -1 * numpy.sum(differences)
        w = -1 * numpy.dot(features.T, differences)
        bias_sum = bias_sum + b ** 2
        weights_sum = weights_sum + w ** 2
        bias = bias - learning_rate * b / (bias_sum ** 0.5)
        weights = weights - learning_rate * w / (weights_sum ** 0.5 + e)
        if i % 100 == 0:
            labels = numpy.round(labels)
            loss = numpy.sum(differences ** 2)
            accuracy = (train_labels.shape[0] - numpy.sum(numpy.abs(train_labels - labels))) / float(train_labels.shape[0])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model = {'b': bias, 'w': weights}
            print('round %d accuracy = %f, loss = %f' % (i, accuracy, loss))
    return model

def test(model, features):
    labels = sigmoid(-1 * numpy.add(model['b'], numpy.dot(features, model['w'])))
    labels = numpy.round(1 - labels)
    return labels

def main():
    labels = read_labels('Y_train.csv')
    features = read_features('X_train.csv')
    test_features = read_features('X_test.csv')
    # normalize
    features = features.T
    test_features = test_features.T
    for i in range(features.shape[0]):
        features[i], mean, sigma = normalize(features[i])
        test_features[i], _, _ = normalize(test_features[i], mean, sigma)
    features = features.T
    test_features = test_features.T
    # train & test
    model = train(labels, features)
    predict_labels = test(model, test_features)
    with open('result2.csv', 'w') as f:
        f.write('id,label\n')
        for i in range(len(predict_labels)):
            f.write('%d,%d\n' % (i + 1, predict_labels[i]))

if __name__ == '__main__':
    main()