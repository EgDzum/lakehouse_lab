import polars as pl

def build_aggregates(silver_path: str, gold_agg_path: str) -> None:
    df = pl.scan_delta(silver_path)
    
    aggregations = [
        {
            "group_by": ["route", "dep_hour"],
            "output_path": f"{gold_agg_path}/agg_route_mart"
        },
        {
            "group_by": ["Marketing_Airline_Network", "Flight_Number_Marketing_Airline"],
            "output_path": f"{gold_agg_path}/agg_airline_mart"
        }
    ]
    
    for agg_config in aggregations:
        result = (
            df.group_by(agg_config["group_by"])
            .agg(
                pl.col("ArrDelayMinutes").mean().alias("avg_arr_delay"),
                pl.col("DepDelayMinutes").mean().alias("avg_dep_delay")
            )
            .collect()
        )
        result.write_delta(agg_config["output_path"], mode="overwrite")

def build_feature_table(silver_path: str, gold_feat_path: str, n: int) -> None:
    path = gold_feat_path
    df = pl.scan_delta(silver_path)
    encoded = (
        df.select(
            ["ArrDelayMinutes", "Distance", "dep_hour", 
             "day_of_week", "Flight_Number_Marketing_Airline", 'Origin']
            )
        .with_columns(
            pl.when(pl.col('ArrDelayMinutes') >= n)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('is_delayed')
            )
        )
    encoded.collect().write_delta(path, mode="overwrite")

if __name__ == "__main__":
    silver_path = './storage/silver'
    gold_agg_path = './storage/gold/agg_mart'
    gold_feat_path = './storage/gold/ml_mart'

    build_aggregates(silver_path, gold_agg_path)

    build_feature_table(silver_path, gold_feat_path, 15)
