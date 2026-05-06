from bronze import BronzeProcessor
from silver import SilverProcessor, optimize_zorder, vacuum_old
from gold import build_aggregates, build_feature_table
from run_ml import FlightDelayModel

import warnings

import mlflow


def main():
    # 1. Bronze: загрузка с батчами по 7 дней (неделями)
    csv_path = './raw_data/flight_data_2018_2024.csv'
    bronze_path = './storage/bronze'
    bronze_processor = BronzeProcessor(
        csv_path=csv_path,
        delta_path=bronze_path,
        batch_size_days=7
    )

    # Запуск Bronze обработки
    bronze_processor.process()

    silver_path = './storage/silver'
    silver_processor = SilverProcessor(
        bronze_path=bronze_path,
        silver_path=silver_path,
        partition_cols=['FlightDate'] 
    )
    # 2. Запуск Silver обработки
    silver_processor.process()

    # 3. Оптимизация
    optimize_zorder(silver_path)
    vacuum_old(silver_path)

    # 4. Gold: агрегаты и feature table
    gold_agg_path = './storage/gold/agg_mart'
    gold_feat_path = './storage/gold/ml_mart'
    build_aggregates(silver_path, gold_agg_path)
    build_feature_table(silver_path, gold_feat_path, 15)

    # 5. ML
    # XGBoost classifier
    mlflow.set_experiment("flight_delay_logreg")
    # чтобы перейти к задаче регрессии нужно поменять аргумент ml_task
    # ml_task имеет два значения: 'reg' и 'cls'
    xgb_model = FlightDelayModel(test_size=0.2, random_state=42, ml_task='cls')

    xgb_model.load_data('./storage/gold/ml_mart')
    xgb_model.split_data()
    xgb_model.create_model()
    xgb_model.train()

    xgb_importance = xgb_model.get_feature_importance()
    # нам нужно закончить трекинг предыдущего эксперимента
    mlflow.end_run()

    # Ridge regression
    mlflow.set_experiment("flight_delay_ridge")
    ridge_model = FlightDelayModel(test_size=0.2, random_state=42, ml_task='reg')

    ridge_model.load_data('./storage/gold/ml_mart')
    ridge_model.split_data()
    ridge_model.create_model()
    ridge_model.train()

    ridge_importance = ridge_model.get_feature_importance()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()