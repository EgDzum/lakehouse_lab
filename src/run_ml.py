import pandas as pd
import polars as pl
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    mean_squared_error, 
    r2_score, 
    mean_absolute_error)
from sklearn.linear_model import Ridge

from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings

warnings.filterwarnings('ignore')


class FlightDelayModel:
    def __init__(self, test_size=0.2, random_state=42, ml_task: str = "cls"):
        self.test_size = test_size
        self.random_state = random_state
        self.ml_task = ml_task

        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

        self.numerical_cols = []
        self.categorical_cols = []

        self.classification_results = {}

    def load_data(self, path):
        """
        Загрузка данных (ожидается parquet/csv)
        """
        # Пример — замените под свой формат
        self.df = pl.scan_delta(path).collect().to_pandas()
        
        if self.ml_task == "cls":
            target = 'is_delayed'
            self.X = self.df.drop(columns=[target, 'ArrDelayMinutes'])
            self.y = self.df[target]
        else:
            target = 'ArrDelayMinutes'
            self.X = self.df.drop(columns=['is_delayed', target])
            self.y = self.df[target]

        # Разделение колонок
        self.numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y if self.ml_task == 'cls' else None
        )

    def create_model(self):
        """
        Создание пайплайна
        """

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )

        if self.ml_task == "cls":
            model = XGBClassifier(
                # n_estimators=100,
                # max_depth=6,
                # learning_rate=0.1,
                # random_state=self.random_state
                )
        else:
            model = Ridge(random_state=self.random_state)

        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])

    def train(self):
        """
        Обучение + MLflow логирование
        """
        if self.model is None:
            raise ValueError("Сначала вызовите create_model()")

        if self.ml_task == 'cls':
            mlflow.xgboost.autolog()
        else:
            mlflow.sklearn.autolog()

        with mlflow.start_run():
            # Параметры
            if self.ml_task == 'cls':
                mlflow.log_param("model", "XGBoost")
            else:
                mlflow.log_param("model", "Ridge")

            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_state", self.random_state)

            # Обучение
            if self.ml_task == 'cls':
                # self.model.fit(self.X_train, self.y_train)
                param_grid = {
                    'model__n_estimators': [100, 150, 200, 300],
                    'model__max_depth': [4, 5, 6],
                    'model__use_label_encoder': [False],
                    'model__random_state': [self.random_state]
                    }
                xgb_grid = GridSearchCV(
                    self.model,
                    param_grid,
                    cv=5,
                    scoring='neg_log_loss',
                    n_jobs=-1,
                    verbose=1
                )
                xgb_grid.fit(self.X_train, self.y_train)
            else:
                param_grid = {
                    'model__alpha': [0.1, 0.5, 1, 10, 100],
                    }

                ridge_grid = GridSearchCV(
                    self.model,
                    param_grid,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                ridge_grid.fit(self.X_train, self.y_train)
            
            if self.ml_task == 'cls':
                self.model = xgb_grid.best_estimator_
                y_pred = self.predict(self.X_test)
                y_pred_proba = self.predict_proba(self.X_test)

                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
                }

                self.classification_results['XGBoost'] = metrics
            else: 
                self.model = ridge_grid.best_estimator_
                y_pred = self.model.predict(self.X_test)

                metrics = {
                    'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                    'MAE': mean_absolute_error(self.y_test, y_pred),
                    'R2': r2_score(self.y_test, y_pred)
                }

                self.classification_results['Ridge'] = metrics

            # Логирование
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.sklearn.log_model(self.model, "model")

            print("\nРЕЗУЛЬТАТЫ XGBOOST:")
            for k, v in metrics.items():
                print(f"{k.upper()}: {v:.4f}")
        
        return self.model

    def predict(self, X=None):
        if X is None:
            X = self.X_test
        return self.model.predict(X)

    def predict_proba(self, X=None):
        if X is None:
            X = self.X_test
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Сначала обучите модель")

        model = self.model.named_steps['model']
        
        if self.ml_task == 'cls':
            importance_df = pd.DataFrame({
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            abs_coefs = np.abs(model.coef_)
            importance_normalized = abs_coefs / abs_coefs.sum()

            importance_df = pd.DataFrame({
                'importance': importance_normalized
            }).sort_values('importance', ascending=False)

        print("\nTOP-10 FEATURES:")
        print(importance_df.head(10))

        # логируем как артефакт
        importance_df.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        return importance_df

if __name__ == "__main__":

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("flight_delay_ridge")

    model = FlightDelayModel(test_size=0.2, random_state=42, ml_task='cls')

    model.load_data('./storage/gold/ml_mart')
    model.split_data()
    model.create_model()
    model.train()

    importance = model.get_feature_importance()
