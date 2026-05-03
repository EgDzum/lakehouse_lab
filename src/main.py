def main():
    # 1. Bronze: загрузка с батчами по 7 дней (неделями)
    csv_path = './raw_data/flight_data_2018_2024.csv'
    bronze_path = './bronze-files/'
    load_bronze_by_days(csv_path, bronze_path)

    # 2. Silver: очистка и партиции по дням
    silver_path = './silver-files'
    create_silver(bronze_path, silver_path)

    # # 3. Оптимизация
    optimize_zorder(silver_path)
    vacuum_old(silver_path)

    # # 4. Gold: агрегаты и feature table
    gold_path = './gold-marts'
    build_aggregates(silver_path, gold_path)
    build_feature_table(silver_path, gold_path, 15)

    # # 5. ML
    # run_ml("./gold-marts/ml_data")

if __name__ == "__main__":
    main()