import polars as pl
import datetime as dt
import sf_quant.data as sfd


def load_data() -> pl.DataFrame:
    """
    Load and prepare market data for signal creation.
    
    Returns:
        pl.DataFrame: Market data with required columns
    """
    # TODO: Load data from source (API, file, database)
    start = dt.date(1996, 1, 1)
    end = dt.date(2024, 12, 31)

    columns = [
        'date',
        'barrid',
        'ticker',
        'price',
        'return',
        'specific_risk',
        'predicted_beta'
    ]

    return sfd.load_assets(
        start=start,
        end=end,
        in_universe=True,
        columns=columns
    ).filter(
        (pl.col('price')
        .shift(1)
        .gt(5))
    )
    
    # TODO: Filter data as needed (date range, symbols, quality checks)
    
    pass

def create_signal(output_path: str = "data/signal.parquet"):
    """
    Loads data, creates a simple signal, and saves it to parquet.
    """
    # TODO: Load Data
    df = load_data()

    # TODO: Add your signal logic here
    signal = (
        df
        .sort('barrid', 'date')
        .with_columns(
            pl.col('return')
            .log1p()
            .rolling_sum(window_size=230)
            .shift(22)
            .alias('signal')
        ).filter(
            pl.col('date') >= dt.date(1997, 1, 1) & pl.col('signal').is_not_null()
        ).with_columns(
            pl.col('specific_risk')
            .fill_null(strategy='forward')
            .over('barrid')
        )
        .with_columns(
            pl.col('signal')
            .sub(pl.col('signal').mean())
            .truediv(pl.col('signal').std())
            .over('date')
            .alias('score')
        )
        .with_columns(
            pl.lit(0.05).mul('score').mul('specific_risk').alias('alpha')
        )
    )
    # TODO: Save to data/signal.parquet
    pl.write_parquet(signal, output_path)

if __name__ == "__main__":
    create_signal()
