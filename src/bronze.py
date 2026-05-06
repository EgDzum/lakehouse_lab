from datetime import datetime
import polars as pl
from deltalake import DeltaTable, write_deltalake

class BronzeProcessor:
    """
    Класс для обработки CSV данных и сохранения их в Delta Lake формате
    с разбиением на батчи по дням.
    """
    
    def __init__(self, csv_path: str, delta_path: str, batch_size_days: int = 7):
        self.csv_path = csv_path
        self.delta_path = delta_path
        self.batch_size_days = batch_size_days
        self._full_df = None
        self._unique_days = None
        
    def _load_csv(self) -> None:
        """Загружает CSV файл и преобразует даты."""
        self._full_df = pl.read_csv(self.csv_path)
        self._full_df = self._full_df.with_columns(
            pl.col("FlightDate").str.strptime(pl.Date, "%Y-%m-%d").alias("Date")
        )
    
    def _get_unique_days(self) -> None:
        """Получает отсортированный список уникальных дней."""
        if self._full_df is None:
            self._load_csv()
            
        self._unique_days = (
            self._full_df.select("Date")
            .unique()
            .sort("Date")
            .to_series()
            .to_list()
        )
    
    def _create_batches(self):
        """Генерирует батчи из уникальных дней."""
        if self._unique_days is None:
            self._get_unique_days()
            
        for i in range(0, len(self._unique_days), self.batch_size_days):
            batch_days = self._unique_days[i:i + self.batch_size_days]
            yield batch_days
     
    def _save_batch(self, batch_df: pl.DataFrame) -> None:
        """Сохраняет батч в Delta Lake."""
        if not DeltaTable.is_deltatable(self.delta_path):
            write_deltalake(self.delta_path, batch_df, mode="overwrite")
        else:
            write_deltalake(self.delta_path, batch_df, mode="append")
    
    def process(self) -> None:
        """
        Основной метод для обработки данных.
        Загружает данные, разбивает на батчи и сохраняет в Delta Lake.
        """        
        # Загружаем данные
        self._load_csv()
        
        # Получаем уникальные дни
        self._get_unique_days()
        
        for batch_days in self._create_batches():
            batch_df = self._full_df.filter(pl.col("Date").is_in(batch_days))
            self._save_batch(batch_df)
       
    
    def get_summary(self) -> dict:
        """
        Возвращает краткую информацию о данных.
        """
        if self._full_df is None:
            self._load_csv()
            
        if self._unique_days is None:
            self._get_unique_days()
            
        return {
            "batch_size_days": self.batch_size_days,
            "total_rows": len(self._full_df),
            "total_batches": len(self._unique_days) // self.batch_size_days
        }

if __name__ == "__main__":
    # Создание класса для обработки данных
    processor = BronzeProcessor(
        csv_path="./raw_data/flight_data_2018_2024.csv",
        delta_path="./storage/bronze",
        batch_size_days=7
    )
    
    # Запуск обработки
    processor.process()

    # Просмотр сводки
    summary = processor.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    