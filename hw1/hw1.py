#!/usr/local/bin/python3
# coding: utf-8
import re
import sys
import numpy
from io import BytesIO

############# DEFAULT VALUE #############
HOURS = 9
CASE = 'PM2.5'
MAX_ITERATION = 200000
VALIDATION_SIZE = 0
ADAGRAD = True
CASES = [
    'PM2.5', 'PM10', 'O3', 'AMB_TEMP', 'RH'
]
SPECIAL_CASES = [
    'PM2.5', 'PM10'
]
OUT_PATH = 'test.csv'
TEST_PATH = 'test_X.csv'
TRAIN_PATH = 'train.csv'

def root_square_mean_error(correct_labels, predict_labels):
    differences = correct_labels - predict_labels
    rmse = (numpy.sum(differences ** 2) / correct_labels.shape[0]) ** 0.5
    return rmse

def read_data(path, is_test = 0):
    refine_data = []
    with open(path, "rb") as f:
        dataString = f.read().decode('big5').replace('NR', '0').encode('utf-8')
        data = numpy.genfromtxt(BytesIO(dataString), delimiter = ',', dtype = None, skip_header = 1 - is_test)
        for row in data:
            case = row[2 - is_test].decode('utf-8')
            if case not in CASES:
                continue
            if is_test == 1:
                m = int(re.search(r'\d+', row[0].decode('utf-8')).group(0))
            else:
                m = int(re.search(r'\d+\/([\d]+)', row[0].decode('utf-8')).group(1)) - 1
            if m not in refine_data:
                refine_data.append([[]] * (len(CASES) + len(SPECIAL_CASES)))
            if case in SPECIAL_CASES:
                index = len(CASES) + SPECIAL_CASES.index(case)
                refine_data[m][index] = refine_data[m][index] + (numpy.array(list(row.tolist())[3 - is_test:]) ** 2).tolist()
            refine_data[m][CASES.index(case)] = refine_data[m][CASES.index(case)] + list(row.tolist())[3 - is_test :]
        labels = []
        features = []
        index = CASES.index(CASE)
        for month in range(len(refine_data)):
            month_data = refine_data[month]
            for j in range(len(month_data[index]) - HOURS + is_test):
                if is_test == 0:
                    labels.append(month_data[index][j + HOURS])
                features.append([d for i in range(len(month_data)) for d in month_data[i][j : j + HOURS]])
        return numpy.array(labels, dtype = float).reshape((len(labels), 1)), numpy.array(features, dtype = float)
    return None, None

def train(train_labels, train_features):
    # gradient descent
    e = 1e-8
    bias = 0
    i = bias_sum = 0
    learning_rate = 1
    best_rmse = sys.maxsize
    weights_sum = numpy.zeros((train_features.shape[1], 1))
    weights = numpy.array([0.2] * train_features.shape[1] ).reshape(train_features.shape[1], 1)
    while i < MAX_ITERATION:
        i = i + 1
        labels = numpy.add(bias, numpy.dot(train_features, weights))
        differences = (train_labels - labels)
        if ADAGRAD == True:
            b = -2 * numpy.sum(differences)
            w = -2 * numpy.dot(train_features.T, differences)
            bias_sum = bias_sum + b ** 2
            weights_sum = weights_sum + w ** 2
            bias = bias - learning_rate * b/ (bias_sum ** 0.5)
            weights = weights - learning_rate * w / (weights_sum ** 0.5 + e)
        else:   
            weights = weights + learning_rate * 2 * numpy.dot(train_features.T, differences)
            bias = bias + learning_rate * 2 * numpy.sum(differences)
        if (i % 1000) == 0:
            best_rmse = (differences ** 2).mean() ** 0.5
            print('round %d => rmse = %f' % (i, best_rmse))
    return {'b': bias, 'w': weights, 'rmse': best_rmse}

def test(model, test_features, path = None):
    labels = numpy.round(numpy.add(model['b'], numpy.dot(test_features, model['w'])))
    if path is not None:
        with open(path, "w") as f:
            f.write('id,value\n');
            for i in range(len(labels)):
                f.write('id_%d,%f\n' % (i, labels[i][0]))
    return labels
    
def main(is_best = False, use_model = False):
    # read train data
    train_labels, train_features = read_data(TRAIN_PATH)
    if train_labels is None or train_features is None or train_labels.shape[0] != train_features.shape[0]:
        print("read train data error."); 
        exit(-1)
    # read test data
    _, test_features = read_data(TEST_PATH, 1)
    if test_features is None:
        print("read test data error."); 
        exit(-1)
    # shuffle row matrix
    try:
        assert is_best
        row_matrix = numpy.load('matrix_best.npy')
    except:
        row_matrix = numpy.identity(train_features.shape[0])
        numpy.random.shuffle(row_matrix)
        row_matrix = numpy.matrix(row_matrix)
    train_labels = numpy.array(row_matrix * numpy.matrix(train_labels))
    train_features = numpy.array(row_matrix * numpy.matrix(train_features))
    # train
    try:
        assert use_model
        model = {}
        model_best = numpy.load('model_best.npz')
        model['b'] = model_best['b']
        model['w'] = model_best['w']
    except:
        model = train(train_labels[VALIDATION_SIZE :], train_features[VALIDATION_SIZE :])
    # validation     
    if VALIDATION_SIZE > 0:
        labels = test(model, train_features[0 : VALIDATION_SIZE], 'test_v.csv')
        rmse = root_square_mean_error(train_labels[0 : VALIDATION_SIZE], labels)
        return model, row_matrix, rmse
    # test
    labels = test(model, test_features, OUT_PATH)
    return model, row_matrix, 0

if __name__ == '__main__':
    numpy.set_printoptions(threshold = numpy.inf)
    if len(sys.argv) > 3:
        OUT_PATH = sys.argv[3]
    if len(sys.argv) > 2:
        TEST_PATH = sys.argv[2]
    if len(sys.argv) > 1:
        TRAIN_PATH = sys.argv[1]
    model, row_matrix, rmse = main()
