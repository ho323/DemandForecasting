from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

class SecondaryRegression:
    def __init__(self):
        self.models = [
            ('RandomForest', RandomForestRegressor()),
            ('XGBoost', XGBRegressor()),
            ('Gradient Boosting', GradientBoostingRegressor()),
            ('LightGBM', LGBMRegressor()),
            ('Huber Regressor', HuberRegressor()),
            ('K Neighbors Regressor', KNeighborsRegressor()),
            ('Linear Regression', LinearRegression())
        ]
        self.second_model = LinearRegression()

    def fit(self, X_train, y_train):
        model_predictions = []
        for name, model in self.models:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            model_predictions.append(y_pred)
            
        second_data = pd.DataFrame({name: pred for name, pred in zip([name for name, _ in self.models], model_predictions)})
        self.second_model.fit(second_data, y_train)
        second_data

    def predict(self, X_test):
        model_predictions = []
        for name, model in self.models:
            y_pred = model.predict(X_test)
            model_predictions.append(y_pred)
            
        second_data = pd.DataFrame({name: pred for name, pred in zip([name for name, _ in self.models], model_predictions)})
        return self.second_model.predict(second_data)