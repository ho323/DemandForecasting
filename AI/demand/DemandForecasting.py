from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor

class DemandForecastingModel:
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
        self.model_predictions = []
        self.second_model = LinearRegression()

    def fit(self, X_train, y_train):
        for name, model in self.models:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.model_predictions.append(y_pred)
            
        self.second_data = pd.DataFrame({name: pred for name, pred in zip([name for name, _ in self.models], self.model_predictions)})
        self.second_model.fit(self.second_data, y_test)

    def predict(self, X_test):
        return self.second_model.predict(self.second_data)