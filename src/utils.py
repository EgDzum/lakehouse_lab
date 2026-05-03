import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)

# import optuna
import mlflow
import logging
from xgboost.callback import TrainingCallback

import joblib
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Custom callback for logging metrics
class LoggingCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        for metric_name, metric_vals in evals_log['test'].items():
            mlflow.log_metric(f"{metric_name}", metric_vals[-1][0], step=epoch)
        return False

class FlightDelayModel:
    """
    Класс для обучения модели XGBoost для предсказания задержек рейсов
    """
    
    def __init__(self, test_size=0.2, random_state=42, experiment_name="flight_delay_prediction"):
        """
        Инициализация класса
        
        Parameters:
        -----------
        test_size : float
            Размер тестовой выборки
        random_state : int
            Random state для воспроизводимости результатов
        """
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classification_results = {}
        self.experiment_name = experiment_name
        
    def load_data(self, file_path):
        """
        Загрузка данных из файла
        
        Parameters:
        -----------
        file_path : str
            Путь к файлу с данными
            
        Returns:
        --------
        self
        """
        # Загрузка данных (замените на вашу логику загрузки)
        # В оригинале был pl.scan_delta, здесь для примера используем pandas
        data = pl.scan_delta(file_path).collect().to_pandas()
        return self._prepare_data(data)
    
    def _prepare_data(self, data):
        """
        Подготовка данных: разделение на признаки и целевые переменные
        
        Parameters:
        -----------
        data : pd.DataFrame
            Исходные данные
        """
        self.y_reg = data['ArrDelayMinutes']
        self.y_cls = data['is_delayed']
        
        # Identify numerical and categorical columns
        self.numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Remove target columns
        self.numerical_cols = [col for col in self.numerical_cols if col not in ['ArrDelayMinutes', 'is_delayed']]
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.X_cls = data.drop(columns=['is_delayed'])
        
        # Encode categorical columns immediately
        for col in self.categorical_cols:
            le = OneHotEncoder()
            self.X_cls[col] = le.fit_transform(self.X_cls[col])
            self.label_encoders[col] = le
        
        return self
    
    def _create_preprocessor(self):
        """
        Создание препроцессора для обработки признаков
        # """
        # cat_cols_cls = self.X_cls.select_dtypes(include=['object', 'category']).columns.tolist()
        # num_cols_cls = self.X_cls.select_dtypes(include=['int64', 'float64']).columns.tolist()

        self.preprocessor_cls = ColumnTransformer([
            ('num', StandardScaler(), self.numerical_cols)
        ], remainder='passthrough')
        
        # Препроцессор для регрессии (если понадобится)
        # self.preprocessor_reg = ColumnTransformer([
        #     ('num', StandardScaler(), self.numerical_cols),
        #     ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_cols)
        # ])

        
    def split_data(self):
        """
        Разделение данных на обучающую и тестовую выборки
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_cls, self.y_cls, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=self.y_cls
        )
        return self
    
    def train(self, max_depth=5):
        """
        Обучение модели
        """
        self._create_preprocessor()

        X_train_processed = self.preprocessor_cls.fit_transform(self.X_train)
        
        feature_names = self.numerical_cols + self.categorical_cols
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)

        # Параметры модели по умолчанию
        params = {
            'n_estimators': [50, 100, 150],
            'max_depth': max_depth,
            'learning_rate': [0.1, 0.01, 0.2],
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # # Создаем пайплайн
        # self.model = Pipeline([
        #     ('preprocessor', self.preprocessor_cls),
        #     ('classifier', xgb.XGBClassifier(**params))
        # ])

        with mlflow.start_run():
            mlflow.log_params(params)
            params.update(eval_metric=['auc', 'error'])
            dtrain = xgb.DMatrix(X_train_processed, label=self.y_train)
            cv_results = xgb.cv(
                params=params,
                dtrain=dtrain,
                num_boost_round=200,
                nfold=3,
                callbacks=[LoggingCallback()],
                verbose_eval=False,
            )

            error = cv_results['test-error-mean'].iloc[-1]
            mlflow.log_metric("accuracy", (1 - error))
            logger.info(f"Attempt: {trial.number}, Accuracy: {1 - error}")

            return error
    
    def predict(self, X=None):
        """
        Предсказание классов
        
        Parameters:
        -----------
        X : pd.DataFrame, optional
            Данные для предсказания. Если не указаны, используются тестовые данные
        """
        if X is None:
            X = self.X_test
        
        return self.model.predict(X)
    
    def predict_proba(self, X=None):
        """
        Предсказание вероятностей
        
        Parameters:
        -----------
        X : pd.DataFrame, optional
            Данные для предсказания. Если не указаны, используются тестовые данные
        """
        if X is None:
            X = self.X_test
        
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self):
        """
        Оценка модели на тестовых данных
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала вызовите create_model() и train()")
        
        # Предсказания
        y_pred = self.predict()
        y_pred_proba = self.predict_proba()
        
        # Метрики
        self.classification_results['XGBoost'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        # Вывод результатов
        print()
        print("РЕЗУЛЬТАТЫ МОДЕЛИ XGBOOST")
        for metric, value in self.classification_results['XGBoost'].items():
            print(f"{metric.upper()}: {value:.4f}")
        print()

        return self.classification_results
    
    def get_feature_importance(self):
        """
        Получение важности признаков
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала вызовите create_model() и train()")
        
        # Получаем обученный классификатор из пайплайна
        xgb_model = self.model.named_steps['classifier']
        
        # Получаем имена признаков после препроцессинга
        # Для OneHotEncoder нужно получить имена созданных признаков
        preprocessor = self.model.named_steps['preprocessor']
        
        # Собираем имена всех признаков
        feature_names = []
        
        # Числовые признаки
        feature_names.extend(self.numerical_cols)
        
        # Категориальные признаки (после one-hot encoding)
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_features = cat_transformer.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_features)
        
        # Создаем DataFrame с важностью признаков
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nТОП-10 НАИБОЛЕЕ ВАЖНЫХ ПРИЗНАКОВ:")
        print(importance_df.head(10))
        
        return importance_df
    
    # def save_model(self, filepath):
    #     """
    #     Сохранение модели
        
    #     Parameters:
    #     -----------
    #     filepath : str
    #         Путь для сохранения модели
    #     """
    #     joblib.dump(self.model, filepath)
    #     print(f"Модель сохранена в {filepath}")
    
    # def load_model(self, filepath):
    #     """
    #     Загрузка модели
        
    #     Parameters:
    #     -----------
    #     filepath : str
    #         Путь к сохраненной модели
    #     """
    #     import joblib
    #     self.model = joblib.load(filepath)
    #     print(f"Модель загружена из {filepath}")
    #     return self


# Пример использования:
if __name__ == "__main__":
    # Создание экземпляра класса
    model = FlightDelayModel(test_size=0.2, random_state=42)
    
    # Загрузка и подготовка данных
    model.load_data('./storage/gold/ml_mart')  # Раскомментируйте для реальных данных
    
    # Разделение данных
    model.split_data()
    
    # Создание и обучение модели
    model.train()
    
    # Оценка модели
    results = model.evaluate()
        
    # Важность признаков
    # importance = model.get_feature_importance()