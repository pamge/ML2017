#!/usr/local/bin/python3
import sys
import numpy as np
import keras.backend as K
from io import BytesIO
from keras import regularizers
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Input, Add, Dot

def load_data(path,mode=0):
    with open(path, "rb") as f:
        dataString = f.read()
        if mode == 0 :
            usecols = (1, 2, 3)
        else:
            usecols = (1, 2)
        data = np.genfromtxt(BytesIO(dataString), delimiter = ',',dtype = np.int64 ,usecols = usecols, skip_header = 1)
    return data

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(((y_true - y_pred) ** 2)))

def gen_model(n_users, n_items, latent_dim = 20):
    user_input = Input(shape = [1])
    item_input = Input(shape = [1])
    # user
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer = 'uniform')(user_input)
    user_vec = Flatten()(user_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer = 'zeros')(user_input)
    user_bias = Flatten()(user_bias)
    # item
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer = 'uniform')(item_input)
    item_vec = Flatten()(item_vec)
    item_bias = Embedding(n_items, 1, embeddings_initializer = 'zeros')(item_input)
    item_bias = Flatten()(item_bias)
    # use mf
    r_hat = Dot(axes = 1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    # use dnn
    # merge_vec = Concatenate()([user_vec, item_vec])
    # hidden = Dense(25, activation='relu')(merge_vec)
    # hidden = Dense(15, activation='relu')(hidden)
    # hidden = Dense(5, activation='relu')(hidden)
    # r_hat = Dense(1)(hidden)
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input, item_input], r_hat)
    model.compile(loss = rmse, optimizer = 'adam')
    return model

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_data = X[indices]
    Y_data = Y[indices]
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]
    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]
    return (X_train,Y_train),(X_val,Y_val)

def main():
    # load data
    data = load_data(sys.argv[1] + './train.csv')
    x_test = load_data(sys.argv[1] + 'test.csv', mode = 1) - 1
    x_train = data.T[0:2].T - 1
    y_train = data.T[2]

    # split data
    (X_train, Y_train), (X_val, Y_val) = split_data(x_train, y_train, split_ratio = 0.10)
    # normalization
    train_std = np.std(Y_train)
    train_mean = np.mean(Y_train)
    Y_val = (Y_val - train_mean) / train_std
    Y_train = (Y_train - train_mean) / train_std

    # train
    # n_users, n_items = 0, 0
    # for sub in data:
    #     if n_users < sub[0]:
    #         n_users = sub[0]
    #     if n_items < sub[1]:
    #         n_items = sub[1]
    # model = gen_model(n_users, n_items, latent_dim = 12)
    # earlystopping = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1, mode = 'min')
    # model.summary()
    # model.fit([X_train.T[0].T, X_train.T[1].T], Y_train,
    #     validation_data = ([X_val.T[0].T, X_val.T[1].T],Y_val),
    #     epochs = 10,
    #     batch_size = 2 << 10,
    #     callbacks = [earlystopping]
    # )
    # model.save('best_model.h5')

    # predit
    model = load_model('best_model.h5', custom_objects = {'rmse': rmse})
    temp_pred =  model.predict([x_test.T[0].T, x_test.T[1].T], batch_size = 2 << 10)
    pred = temp_pred * train_std + train_mean
    with open(sys.argv[2], "w") as f:
        f.write("TestDataID,Rating\n")
        for idx,p in enumerate(pred ,1):
            for val in p:
                f.write("{},{}\n".format(idx,val))

if __name__ == '__main__':
    main()
