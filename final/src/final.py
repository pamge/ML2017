import sys
import numpy as np
import pandas as pd
import xgboost as xgb

XGP_PARAMETERS = {
    'eta': 0.05,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0
}

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    # check argv
    if len(sys.argv) != 5:
        print('usage: ./python3 final.py train.csv macro.csv test.csv i_love_syaro.csv')
        exit(0)
    # read argv
    TRAIN_PATH = sys.argv[1] # './train.csv'
    MACRO_PATH = sys.argv[2] # './macro.csv'
    TEST_PATH = sys.argv[3] # './test.csv'
    OUT_PATH = sys.argv[4] # 'i_love_syaro.csv'
    MODEL_PATH = 'model'

    # read csv
    train_data = pd.read_csv(TRAIN_PATH, parse_dates = ['timestamp'])
    test_data = pd.read_csv(TEST_PATH, parse_dates = ['timestamp'])
    macro_data = pd.read_csv(MACRO_PATH, parse_dates = ['timestamp'])
    
    # concat training data and testing data (with some base preprocessing)
    ID_test = test_data['id']
    Y_train = train_data['price_doc'].values

    test_data = test_data.drop(['id'], axis = 1)
    train_data = train_data.drop(['id', 'price_doc'], axis = 1)

    data = pd.concat([train_data, test_data])
    
    # Preprocess 1: data.join(macro_data)
    data = data.join(macro_data, on = 'timestamp', rsuffix = '_macro')
    
    # Preprocess 2: if life_sq > full_sq, then life_sq = full_sq * 0.6
    for index in data[data.life_sq > data.full_sq].index:
        data['life_sq'][index] = data['full_sq'][index] * 0.6
    
    # Preprocess 3: if kitch_sq > life_sq, then kitch_sq = life_sq * 0.25
    for index in data[data.kitch_sq > data.life_sq].index:
        data['kitch_sq'][index] = data['life_sq'][index] * 0.25
               
    # Preprocess 4: if build_year < 1910 or build_year > 2017, then build_year = NaN
    data.ix[data[data.build_year < 1910].index, 'build_year'] = np.NaN
    data.ix[data[data.build_year > 2017].index, 'build_year'] = np.NaN
           
    # Preprocess 5: if floor <= 0 or max_floor <= 0, then floor = NaN, max_floor = NaN
    data.ix[data[(data.floor <= 0) | (data.max_floor <= 0)].index, ['floor', 'max_floor']] = np.NaN
           
    # Preprocess 6: if floor > max_floor, then floor = max_floor * 0.6
    for index in data[data.floor > data.max_floor].index:
        data['floor'][index] = data['max_floor'][index] * 0.6
    
    # Preprocess 7: month, month_cnt = process_timestamp(timestamp)
    data['month'] = data.timestamp.dt.month        
    month_year = (data.timestamp.dt.month + data.timestamp.dt.year * 12)
    data['month_cnt'] = month_year.map(month_year.value_counts().to_dict())
    
    # Preprocess 8: floor_rate = floor / max_floor
    data['floor_rate'] = data['floor'] / data['max_floor'].astype(float)
    
    # Preprocess 9: kitch_rate, life_rate = kitch_sq / full_sq, life_sq / full_sq
    data['rel_kitch_sq'] = data['kitch_sq'] / data['full_sq'].astype(float)
    data['rel_life_sq'] = data['life_sq'] / data['full_sq'].astype(float)
    
    # Preprocess 10: data = process_category(data)
    data_numeric = data.select_dtypes(exclude = ['object'])
    data_string = data.select_dtypes(include = ['object']).copy()
    for c in data_string:
        data_string[c] = pd.factorize(data_string[c])[0]        
    data = pd.concat([data_numeric, data_string], axis = 1)
    
    # split to training data and testing data (with some base preprocessing)
    data = data.drop(['timestamp', 'timestamp_macro'], axis = 1)
    X_train = data.values[:len(train_data)]
    X_test = data.values[len(train_data):]

    # train or load model
    """
    model = xgb.train(dict(XGP_PARAMETERS), xgb.DMatrix(X_train, Y_train), num_boost_round = 250)
    model.save_model('model')
    """
    model = xgb.Booster({'nthread':4})
    model.load_model(MODEL_PATH)
    
    # predict result
    Y_test = model.predict(xgb.DMatrix(X_test))
    result = pd.DataFrame({'id': ID_test, 'price_doc': Y_test})
    result.to_csv(OUT_PATH, index = False)