#!/usr/local/bin/python3
# coding: utf-8
import re
import sys
import numpy
from io import BytesIO

############# DEFAULT VALUE #############
HOURS = 9
CASE = 'PM2.5'
MAX_ITERATION = 100000
VALIDATION_SIZE = 240
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
    # matrix
    train_features2 = numpy.concatenate((train_features, numpy.array([[1] * train_features.shape[0]]).T), axis = 1)
    Y = numpy.matrix(train_labels)
    X = numpy.matrix(train_features2)
    weights = numpy.array(numpy.linalg.inv(X.T * X) * X.T * Y)
    model = {'b': numpy.sum(weights[:][-1]), 'w': weights[:][0 : -1]}
    return model, root_square_mean_error(train_labels, numpy.dot(train_features2, weights))

def test(model, test_features, path = None):
    labels = numpy.round(numpy.add(model['b'], numpy.dot(test_features, model['w'])))
    if path is not None:
        with open(path, "w") as f:
            f.write('id,value\n');
            for i in range(len(labels)):
                f.write('id_%d,%f\n' % (i, labels[i][0]))
    return labels
    
def main(use_index = False, use_model = False):
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
        assert use_index
        index = numpy.load('index_best').tolist()
    except:
        index = numpy.random.permutation(train_labels.shape[0]).tolist()
    temp_train_labels = []
    temp_train_features = []
    for i in index:
        temp_train_labels.append(train_labels[i])
        temp_train_features.append(train_features[i])
    train_labels = numpy.array(temp_train_labels)
    train_features = numpy.array(temp_train_features)
    # train
    try:
        assert use_model
        model = {}
        model_best = numpy.load('model_best')
        model['b'] = model_best['b']
        model['w'] = model_best['w']
        train_rmse = 0
    except:
        model, train_rmse = train(train_labels[VALIDATION_SIZE :], train_features[VALIDATION_SIZE :])
    # test
    labels = test(model, test_features, OUT_PATH)
    # validation     
    if VALIDATION_SIZE > 0:
        labels = test(model, train_features[0 : VALIDATION_SIZE], 'test_v.csv')
        rmse = root_square_mean_error(train_labels[0 : VALIDATION_SIZE], labels)
        return model, index, rmse, train_rmse
    return model, index, 0, train_rmse

if __name__ == '__main__':
    numpy.set_printoptions(threshold = numpy.inf)
    if len(sys.argv) > 3:
        OUT_PATH = sys.argv[3]
    if len(sys.argv) > 2:
        TEST_PATH = sys.argv[2]
    if len(sys.argv) > 1:
        TRAIN_PATH = sys.argv[1]
    # tune parameters
    # best_rmse = sys.maxsize
    # for i in range(500):
    #     model, index, rmse, train_rmse = main()
    #     if train_rmse < best_rmse:
    #         best_rmse = train_rmse
    #         numpy.save('index.npy', index)
    #         numpy.savez('model.npz', b = model['b'], w = model['w'])
    #     print('i = %d, rmse = %f, train_rmse = %f' % (i + 1, rmse, train_rmse))
    # print('best rmse => %f' % best_rmse)
    model, index, rmse, train_rmse = main(True, True)
    print('rmse = %f, train_rmse = %f' % (rmse, train_rmse))
