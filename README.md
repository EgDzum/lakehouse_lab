## Запуск

Чтобы запустить проект, перейдите в директорию репозитория и выполните следующую команду
```bash
docker compose up
```

- После запуска контейнеров результат обучения моделей можно посмотреть в интерфейсе mlflow (https://localhost:5000). 
- Хранилищем выступают локальные директории (./data локально и ./storage в контейнере).
- В датасете содержатся данные только за один месяц, поэтому на этапе silver мы партицируем по дням.

## Структура проекта

## Bronze

- Загружаем данные с помощью абстракции ```BronzeProcessor```.
- Партицируем данные за неделю, чтобы уменьшить количество партиций.
```python
 def _create_batches(self):
        """Генерирует батчи из уникальных дней."""
        if self._unique_days is None:
            self._get_unique_days()
            
        for i in range(0, len(self._unique_days), self.batch_size_days):
            batch_days = self._unique_days[i:i + self.batch_size_days]
            yield batch_days
```
- Загружаем данные в режиме append
```python
write_deltalake(self.delta_path, batch_df, mode="append")
```

## Silver 

- Создаем обработчик данных ```SilverProcessor```.
- Фильтруем данные (избавляемся от выбросов, противоречивых данных) и добавляем новые признаки.
- Делаем Merge с существующей таблицей. Если такой нет, то создаем новую.
- Оптимизируем хранение данных с помощью 
  - ```optimize.compact()``` - перезаписывает множество мелких файлов в более крупные.
  - ```dt.vacuum()``` удаление файлов данных, которые не участвуют в текущей версии таблицы и попадают в «историю» старше указанного времени. 
  - ```.optimize.zorder(['route'])``` - упорядочивает данные так, чтобы строки с похожими значениями в указанных столбцах лежали в одних и тех же файлах.

## Gold

Создаем две витрины: 
- Группировка по 1) ["route", "dep_hour"] и 2) ["Marketing_Airline_Network", "Flight_Number_Marketing_Airline"]. В обоих таблицах мы считаем средние значения задержки при прилёте (```pl.col("ArrDelayMinutes")```) и позднего отправления из аэропорта (```pl.col("DepDelayMinutes")```).
- ML таблица с необходимыми столбцами + добавление нового признака ```is_delayed```

## ML

- Для классификации мы используем XGBoost. Target - ```is_delayed```
- Для задачи регрессии мы используем линейную регрессию с l2 регуляризацией (Ridge). Target - ```"ArrDelayMinutes"```
- cравнение моделей можно найти в ноутбуке в директории ./notebooks/
- процесс логгирования обучения, параметров модели и выборки, а также важность признаков можно увидеть в mlflow

## План запроса .explain()

Посмотрим план запроса, который используется для создания новых признаков.

```python
clean_df = (
        bronze
        .filter(pl.col("Cancelled") == 0)
        .filter(pl.col("DepTime") != "")
        .filter(pl.col("ArrTime") != "")
        .filter(pl.col("DepDelayMinutes").is_not_null())
        .filter(pl.col("ArrDelayMinutes").is_not_null())
        .with_columns([
            pl.col("Date").dt.weekday().alias("day_of_week"),
            (pl.col("Origin") + pl.col("Dest")).alias("route"),
        ])
        .with_columns(
            pl.when(pl.col("DepTime") == "2400")
            .then(pl.lit(0))
            .otherwise(
                pl.col("DepTime").str.slice(0, 2).cast(pl.Int32)
            )
            .alias("dep_hour")
        )
        .select(["FlightDate", "Flight_Number_Marketing_Airline",
                 "ArrDelayMinutes", "DepDelayMinutes", "Distance",
                 "dep_hour", "day_of_week", "route",
                 "Year", "Origin", "Marketing_Airline_Network"])
    )
```

Результат выглядит следующим образом:

```
simple π 11/11 ["FlightDate", ... 10 other columns]

WITH_COLUMNS:
  [col("Date").dt.weekday().alias("day_of_week"),
   col("Origin").str.concat_horizontal([col("Dest")]).alias("route"),
   when(col("DepTime") == "2400")
     .then(0)
     .otherwise(col("DepTime").str.slice([dyn int: 0, dyn int: 2]).strict_cast(Int32))
     .alias("dep_hour")]

Parquet SCAN [/content/delta-files/part-00000-8669183a-942a-48be-b189-fa6e210ae705-c000.snappy.parquet, ... 9 other sources]

PROJECT 13/121 COLUMNS

SELECTION:
  [([([([([(col("DepTime")) != ("")]) & ([(col("Cancelled")) == (0.0)])]) & (col("DepDelayMinutes").is_not_null())]) & (col("ArrDelayMinutes").is_not_null())]) & ([(col("ArrTime")) != ("")])]
```

Что мы видим:
- simple π 11/11 ["FlightDate", ...] — на выходе нужен только набор из 11 колонок, то есть лишние столбцы в итог не попадут

- WITH_COLUMNS — Polars добавляет новые столбцы поверх уже отфильтрованных данных, здесь это day_of_week, route, dep_hour

- Parquet SCAN [...] — данные читаются не целиком в память, а через ленивое чтение

- PROJECT 13/121 COLUMNS — из 121 колонки исходного файла реально читаются только 13, это projection pushdown

- SELECTION: ... — фильтр, который применяется ко строкам
