from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import numpy as np
import pandas as pd

from DemandForecasting import SecondaryDemandForecastingModel


def testData(data_path):
    # Load data
    data = pd.read_csv(data_path)

    df = data[['store_id', 'order_div', 'order_dt', 'menu_name', 'sale_price', 'quentity']]

    # 취소된 주문 삭제
    idx = df[df['order_div'] == '취소'].index
    df = df.drop(idx)
    df = df.drop('order_div',axis=1)

    # menu_name 필요 없는 데이터 삭제
    df = df[df.menu_name != '추가배달료 결제 감사합니다']
    df = df[df.menu_name != '코카콜라']
    df = df[df.menu_name != '사이다']

    # menu_name 숫자형으로 변환
    mapping = {}
    for i, j in enumerate(df['menu_name'].unique()):
        mapping[j] = i
        
    df.loc[:,'menu_name'] = df.loc[:,'menu_name'].map(mapping)
    df['menu_name'] = df['menu_name'].astype(int)

    # 월 별로 묶음
    df['order_dt'] = pd.to_datetime(df['order_dt'], format='%Y%m%d')
    df['year'] = df['order_dt'].dt.year
    df['month'] = df['order_dt'].dt.month
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df = df.drop('order_dt', axis=1)

    # 계절
    df['season'] = df['month'].apply(lambda x: 1 if x in [3, 4, 5] else (2 if x in [6, 7, 8] else (3 if x in [9, 10, 11] else 4)))

    # 합산
    df = df.groupby(['store_id', 'menu_name', 'sale_price', 'year', 'month', 'season']).sum().reset_index()

    X = df.drop('quentity', axis=1)
    y = df['quentity']

    # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def multipleError(y_pred, y_test):
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_pred, y_test)
    print("MAE:", mae)

    # MSE (Mean Squared Error)
    mse = mean_squared_error(y_pred, y_test)
    print("MSE:", mse)

    # RMSE (Root Mean Squared Error)
    rmse = mean_squared_error(y_pred, y_test, squared=False)
    print("RMSE:", rmse)

    # R2 (Coefficient of Determination)
    r2 = r2_score(y_pred, y_test)
    print("R2:", r2)

    # MAPE (Mean Absolute Percentage Error)
    def mean_absolute_percentage_error(y_pred, y_test):
        return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    mape = mean_absolute_percentage_error(y_pred, y_test)
    print("MAPE:", mape)

if __name__ == '__main__':
    data_path = "/Users/ho/Documents/lld/order_info_202307111047.csv"
    X_train, X_test, y_train, y_test = testData(data_path)

    model = SecondaryDemandForecastingModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    multipleError(y_pred, y_test)