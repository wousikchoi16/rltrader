import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import settings


COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_TRAINING_DATA_V1 = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V1_RICH = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'inst_lastinst_ratio', 'frgn_lastfrgn_ratio',
    'inst_ma5_ratio', 'frgn_ma5_ratio',
    'inst_ma10_ratio', 'frgn_ma10_ratio',
    'inst_ma20_ratio', 'frgn_ma20_ratio',
    'inst_ma60_ratio', 'frgn_ma60_ratio',
    'inst_ma120_ratio', 'frgn_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V2 = [
    'per', 'pbr', 'roe',
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio', 
    'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio', 
    'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio', 
    'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio'
]

COLUMNS_TRAINING_DATA_V3 = [
    'per', 'pbr', 'roe',
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'diffratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio', 
    'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio', 
    'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio', 
    'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio',
    'ind', 'ind_diff', 'ind_ma5', 'ind_ma10', 'ind_ma20', 'ind_ma60', 'ind_ma120',
    'inst', 'inst_diff', 'inst_ma5', 'inst_ma10', 'inst_ma20', 'inst_ma60', 'inst_ma120',
    'foreign', 'foreign_diff', 'foreign_ma5', 'foreign_ma10', 'foreign_ma20', 'foreign_ma60', 'foreign_ma120',
]

def preprocess(data, ver='v1'): #load_data????????? ??????. 
    windows = [5, 10, 20, 60, 120]
    for window in windows: # ????????? f-string?????? ?????????
        data['close_ma{}'.format(window)] = data['close'].rolling(window).mean()
        data['volume_ma{}'.format(window)] = data['volume'].rolling(window).mean()
        data['close_ma%d_ratio' % window] = (data['close'] - data['close_ma%d' % window]) / data['close_ma%d' % window]
        data['volume_ma%d_ratio' % window] = (data['volume'] - data['volume_ma%d' % window]) / data['volume_ma%d' % window]
            
        if ver == 'v1.rich': # ???????????? ??????(?????????)
            data['inst_ma{}'.format(window)] = data['close'].rolling(window).mean()
            data['frgn_ma{}'.format(window)] = data['volume'].rolling(window).mean()
            data['inst_ma%d_ratio' % window] = (data['close'] - data['inst_ma%d' % window]) / data['inst_ma%d' % window]
            data['frgn_ma%d_ratio' % window] = (data['volume'] - data['frgn_ma%d' % window]) / data['frgn_ma%d' % window]

    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
        (data['volume'][1:].values - data['volume'][:-1].values) 
        / data['volume'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values
    )

    if ver == 'v1.rich':
        data['inst_lastinst_ratio'] = np.zeros(len(data))
        data.loc[1:, 'inst_lastinst_ratio'] = (
            (data['inst'][1:].values - data['inst'][:-1].values)
            / data['inst'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values # replace(to_replace=0, method='ffill'): 0?????? ??????????????? ??????. 
        )
        data['frgn_lastfrgn_ratio'] = np.zeros(len(data))
        data.loc[1:, 'frgn_lastfrgn_ratio'] = (
            (data['frgn'][1:].values - data['frgn'][:-1].values)
            / data['frgn'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values
        )

    return data 
    # columns=['close_ma5','close_ma5_ratio','volume_ma5','volume_ma5_ratio','inst_ma5','inst_ma5_ratio','frgn_ma5','frgn_ma5_ratio',
    #  'open_lastclose_ratio','low_close_ratio','high_close_ratio','close_lastclose_ratio','volume_lastvolume_ratio']


def load_data(code, date_from, date_to, ver='v2'):
    if ver == 'v3': # ?????? ??????
        return load_data_v3(code, date_from, date_to)

    header = None if ver == 'v1' else 0
    data = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data\{}\{}.csv'.format(ver, code)), 
        thousands=',', header=header, converters={'date': lambda x: str(x)}) ##> format(args.ver,-> format(ver,  ##> stock_code->code

    if ver == 'v1':
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # ?????? ???????????? ??????
    data = data.sort_values(by='date').reset_index()

    # ????????? ?????????
    data = preprocess(data)
    
    # ?????? ?????????
    data['date'] = data['date'].str.replace('-', '') # v1????????? ?????????. 
    data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]
    data = data.dropna()

    # ?????? ????????? ??????
    chart_data = data[COLUMNS_CHART_DATA] # ?????? OHLCV (???6??????)

    # ?????? ????????? ??????
    training_data = None
    if ver == 'v1':
        training_data = data[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.rich':
        training_data = data[COLUMNS_TRAINING_DATA_V1_RICH]
    elif ver == 'v2':
        data.loc[:, ['per', 'pbr', 'roe']] = data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception('Invalid version.')
    
    return chart_data, training_data


def load_data_v3(code, date_from, date_to): # ?????? ??????
    df = None
    for filename in os.listdir('C:\projecth\example\rltrader-master\data\v3'): ##>  'D:\\dev\\rltrader\\data\\v3'
        if filename.startswith(code):
            df = pd.read_csv(os.path.join('C:\projecth\example\rltrader-master\data\v3', filename), thousands=',', header=0, converters={'date': lambda x: str(x)}) ##>
            break

    # ?????? ???????????? ??????
    df = df.sort_values(by='date').reset_index()

    # ?????????
    scaler = StandardScaler()
    scaler.fit(df[COLUMNS_TRAINING_DATA_V3].dropna().values)

    # ?????? ?????????
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.dropna()

    # ?????? ????????? ??????
    chart_data = df[COLUMNS_CHART_DATA]

    # ?????? ????????? ??????
    training_data = df[COLUMNS_TRAINING_DATA_V3]
    training_data = pd.DataFrame(scaler.transform(training_data.values), columns=COLUMNS_TRAINING_DATA_V3)
    
    return chart_data, training_data

#????????????
# a,b=load_data('051910', '20170101', '20171231', ver='v2')
# print(b) # ??????(??????????????? ?????? ??????,26)