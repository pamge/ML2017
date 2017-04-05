#!/usr/local/bin/python3
import sys
import math
import numpy

def read_labels(path):
    return numpy.genfromtxt(path, delimiter = ',', dtype = float, skip_header = 0)

def read_features(path):
    return numpy.genfromtxt(path, delimiter = ',', dtype = float, skip_header = 1)
    
def train(labels, features):
    def maximum_likelihood(labels, features):
        mean = numpy.mean(features, axis = 0)
        cov = numpy.cov(features.T)
        return {'mean': mean, 'cov': cov}
    true_indices = numpy.where(labels == 1.0)
    false_indices = numpy.where(labels == 0.0)
    true_size = true_indices[0].shape[0]
    false_size = false_indices[0].shape[0]
    true_model = maximum_likelihood(labels[true_indices], features[true_indices])
    false_model = maximum_likelihood(labels[false_indices], features[false_indices])
    cov = (true_model['cov'] * true_size + false_model['cov'] * false_size) / float(labels.shape[0])
    return {'m1': true_model['mean'], 'm2': false_model['mean'], 'cov': cov, 'm1_size': true_size, 'm2_size': false_size}

def test(model, features):
    k = model['m1'].shape[0]
    inv = numpy.linalg.inv(model['cov'])
    det = numpy.linalg.det(model['cov'])
    p1 = float(model['m1_size']) / (model['m1_size'] + model['m2_size'])
    p2 = float(model['m2_size']) / (model['m1_size'] + model['m2_size'])
    x = 1.0 / (math.pow(2 * numpy.pi, k / 2) * numpy.sqrt(det))
    labels = []
    for feature in features:
        d1 = (feature - model['m1']).reshape(-1, 1)
        d2 = (feature - model['m2']).reshape(-1, 1)
        exp1 = numpy.exp(-0.5 * numpy.dot(numpy.dot(d1.T, inv), d1)[0][0])
        exp2 = numpy.exp(-0.5 * numpy.dot(numpy.dot(d2.T, inv), d2)[0][0])
        if exp1 == 0 and exp2 == 0:
            exp1 = 1
            exp2 = 1
        if (x * exp1 * p1) / (x * exp1 * p1 + x * exp2 * p2) > 0.5:
            labels.append(1)
        else:
            labels.append(0)
    return labels

def main():
    assert(len(sys.argv) == 5)
    # read argv
    X_TRAIN = sys.argv[1]
    Y_TRAIN = sys.argv[2]
    X_TEST = sys.argv[3]
    OUTPUT = sys.argv[4]
    # read data
    labels = read_labels(Y_TRAIN)
    features = read_features(X_TRAIN)
    test_features = read_features(X_TEST)
    model = train(labels, features)
    predict_labels = test(model, test_features)
    with open(OUTPUT, 'w') as f:
        f.write('id,label\n')
        for i in range(len(predict_labels)):
            f.write('%d,%d\n' % (i + 1, predict_labels[i]))
    
if __name__ == '__main__':
    main()