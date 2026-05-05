from deltalake import DeltaTable, write_deltalake
import polars as pl
import pyarrow as pa
from typing import Optional


class SilverProcessor:
    """
    Класс для обработки и записи Silver слоя данных
    
    Parameters
    ----------
    bronze_path : str
        Путь к Bronze таблице (Delta Lake)
    silver_path : str
        Путь для записи Silver таблицы (Delta Lake)
    partition_cols : list, optional
        Колонки для партиционирования, по умолчанию ['FlightDate']
    """
    
    def __init__(self, bronze_path: str, silver_path: str, partition_cols: Optional[list] = None):
        self.bronze_path = bronze_path
        self.silver_path = silver_path
        self.partition_cols = partition_cols or ['FlightDate']
        
    def _filter_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Фильтрация данных от некорректных записей"""
        return (
            df
            .filter(pl.col("Cancelled") == 0)
            .filter(pl.col("DepTime") != "")
            .filter(pl.col("ArrTime") != "")
            .filter(pl.col("DepDelayMinutes").is_not_null())
            .filter(pl.col("ArrDelayMinutes").is_not_null())
        )
    
    def _create_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Создание новых признаков"""
        return (
            df
            .with_columns([
                pl.col("Date").dt.weekday().alias("day_of_week"),
                (pl.col("Origin") + pl.col("Dest")).alias("route"),
            ])
            .with_columns(
                pl.when(pl.col("DepTime") == "2400")
                .then(pl.lit(0))
                .otherwise(pl.col("DepTime").str.slice(0, 2).cast(pl.Int32))
                .alias("dep_hour")
            )
        )
    
    def _select_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Выбор необходимых колонок"""
        return df.select([
            "FlightDate", "Flight_Number_Marketing_Airline",
            "ArrDelayMinutes", "DepDelayMinutes", "Distance",
            "dep_hour", "day_of_week", "route", 
            "Year", "Origin", "Marketing_Airline_Network"
        ])
    
    def _load_bronze_data(self) -> pa.Table:
        """
        Загрузка и обработка Bronze данных
        
        Returns
        -------
        pyarrow.Table
            Обработанные данные в формате Arrow
        """
        bronze = pl.scan_delta(self.bronze_path)
        
        clean_df = (
            bronze
            .pipe(self._filter_data)
            .pipe(self._create_features)
            .pipe(self._select_columns)
            .collect()
            .to_arrow()
        )
        
        return clean_df       
    
    def process(self) -> None:
        """
        Основной метод для выполнения полного цикла обработки:
        1. Загрузка и очистка данных из Bronze
        2. Создание новых признаков
        3. Запись в Silver с партиционированием
        4. Слияние с существующими данными (если таблица существует)
        """
        processed_data = self._load_bronze_data()
                
        try:
            dt = DeltaTable(self.silver_path)
        
            predicate = (
                "s.FlightDate = t.FlightDate AND "
                "s.Flight_Number_Marketing_Airline = t.Flight_Number_Marketing_Airline"
            )
            
            dt.merge(
                processed_data,
                predicate=predicate,
                source_alias="s",
                target_alias="t"
            ).when_not_matched_insert_all().execute()
            
        except Exception as e:
            # Если таблица не существует, создаем новую
            write_deltalake(
                self.silver_path,
                processed_data,
                mode="overwrite",
                partition_by=self.partition_cols
                )
        print("Таблица успешно создана")

def optimize_zorder(dt_table_path: str) -> None:
    dt = DeltaTable(dt_table_path)
    dt.optimize.compact()

def vacuum_old(dt_table_path: str, retention_hours: int = 24) -> None:
    dt = DeltaTable(dt_table_path)
    dt.vacuum(
        retention_hours=retention_hours,
        enforce_retention_duration=False,
        dry_run=False
        )

if __name__ == "__main__":
    # Создание экземпляра процессора
    processor = SilverProcessor(
        bronze_path="./storage/bronze",
        silver_path="./storage/silver",
        partition_cols=['FlightDate'] 
    )
    
    # Запуск обработки
    processor.process()
