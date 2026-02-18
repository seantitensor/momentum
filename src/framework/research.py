import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from sf_quant.schema import AlphaSchema
    import sf_quant.research as sfr
    import polars_ols
    import pandas
    return AlphaSchema, go, marimo, np, pl, sfr


@app.cell
def _(marimo):
    signal_file = marimo.ui.text(
        value="data/signal.parquet",
        label="Signal file path:"
    )
    return (signal_file,)


@app.cell
def _(pl, signal_file):
    signal_df = pl.read_parquet(signal_file.value).filter(
        pl.col('alpha').is_not_null()
    )
    return (signal_df,)


@app.cell
def _(marimo, signal_df):
    if signal_df is None:
        marimo.stop(marimo.md("**⚠️ Please load a valid signal file first**"))
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Signal Analysis

    Examine signal statistics and characteristics before portfolio construction.
    """)
    return


@app.cell
def _(sfr, signal_df):
    sfr.get_signal_stats(signal_df)
    return


@app.cell
def _(sfr, signal_df):
    sfr.get_signal_distribution(signal_df)
    return


@app.cell
def _(marimo):
    n_quantiles = marimo.ui.slider(
        value=5,
        start=2,
        stop=10,
        step=1,
        label="Number of quantiles:"
    )
    return (n_quantiles,)


@app.cell
def _(marimo):
    marimo.md("""
    ## Quantile Portfolio Construction

    Build long-short portfolios based on signal quantiles.
    """)
    return


@app.cell
def _(n_quantiles, sfr, signal_df):
    # Create quantile portfolios based on alpha signal
    quantile_df = sfr.generate_quantile_ports(
        signal_df,
        num_bins=n_quantiles.value,
        signal_col='alpha'
    )

    quantile_df.head(10)
    return (quantile_df,)


@app.cell
def _(n_quantiles, pl, quantile_df):
    # Convert quantile portfolios to long format for performance analysis
    ports_long = (
        quantile_df
        .unpivot(
            index="date",
            variable_name="quantile",
            value_name="return"
        )
    )

    ports_long.head(10)
    return (ports_long,)


@app.cell
def _(marimo):
    include_ff_regression = marimo.ui.checkbox(
        value=True,
        label="Include Fama-French regression analysis"
    )
    return (include_ff_regression,)


@app.cell
def _(marimo):
    marimo.md("""
    ## Portfolio Performance Analysis

    Calculate and visualize performance metrics across quantile portfolios.
    """)
    return


@app.cell
def _(pl, ports_long):
    # Calculate quantile portfolio returns (equal-weighted)
    quantile_perf = (
        ports_long
        .group_by(["date", "quantile"])
        .agg(pl.col("return").mean().alias("portfolio_return"))
        .sort(["date", "quantile"])
    )

    quantile_perf.head(20)
    return (quantile_perf,)


@app.cell
def _(pl, quantile_perf):
    # Calculate cumulative returns for each quantile
    cumul_returns = (
        quantile_perf
        .with_columns([
            (1 + pl.col("portfolio_return")).log().alias("log_return")
        ])
        .with_columns([
            pl.col("log_return").cum_sum().over("quantile").exp().alias("cum_return")
        ])
        .select(["date", "quantile", "cum_return"])
    )

    cumul_returns
    return (cumul_returns,)


@app.cell
def _(cumul_returns, go, n_quantiles, pl):
    # Plot cumulative returns
    fig = go.Figure()

    for i in range(n_quantiles.value):
        quantile = f"p_{i+1}"
        data = cumul_returns.filter(pl.col("quantile") == quantile)
        dates = data.select("date").to_numpy().flatten()
        returns = data.select("cum_return").to_numpy().flatten()

        fig.add_trace(go.Scatter(
            x=dates,
            y=returns,
            mode='lines',
            name=f"Q{i+1}",
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d}<br>Cum Return: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title="Quantile Portfolio Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified',
        height=500,
        template="plotly_white"
    )

    fig.show()
    return


@app.cell
def _(np, pl, quantile_perf):
    # Calculate performance metrics
    metrics = (
        quantile_perf
        .with_columns([
            (1 + pl.col("portfolio_return")).log().alias("log_return")
        ])
        .group_by("quantile")
        .agg([
            pl.col("portfolio_return").mean().alias("mean_return"),
            pl.col("portfolio_return").std().alias("std_return"),
            pl.col("log_return").sum().alias("cum_log_return"),
            pl.col("portfolio_return").count().alias("n_obs"),
        ])
        .with_columns([
            (pl.col("cum_log_return").exp() - 1).alias("total_return"),
            (pl.col("mean_return") / pl.col("std_return")).alias("sharpe_ratio"),
            (pl.col("mean_return") * 252).alias("annual_return"),  # Assuming monthly data
            (pl.col("std_return") * np.sqrt(12)).alias("annual_vol"),
        ])
        .select([
            "quantile",
            "annual_return",
            "annual_vol",
            "sharpe_ratio",
            "total_return",
            "n_obs"
        ])
        .sort("quantile")
    )

    metrics
    return (metrics,)


@app.cell
def _(go, metrics):
    # Plot Sharpe Ratios
    fig_sharpe = go.Figure()

    _quantiles = metrics.select("quantile").to_numpy().flatten()
    _sharpe = metrics.select("sharpe_ratio").to_numpy().flatten()

    fig_sharpe.add_trace(go.Bar(
        x=_quantiles,
        y=_sharpe,
        marker_color='steelblue',
        hovertemplate='<b>%{x}</b><br>Sharpe Ratio: %{y:.3f}<extra></extra>'
    ))

    fig_sharpe.update_layout(
        title="Sharpe Ratio by Quantile",
        xaxis_title="Quantile",
        yaxis_title="Sharpe Ratio",
        height=400,
        template="plotly_white"
    )

    fig_sharpe.show()
    return


@app.cell
def _(go, metrics):
    # Plot Annual Returns vs Volatility
    _quantiles = metrics.select("quantile").to_numpy().flatten()
    _ret = metrics.select("annual_return").to_numpy().flatten()
    _vol = metrics.select("annual_vol").to_numpy().flatten()

    fig_ef = go.Figure()

    fig_ef.add_trace(go.Scatter(
        x=_vol,
        y=_ret,
        mode='markers+text',
        marker=dict(size=12, color='steelblue'),
        text=_quantiles,
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>Annual Vol: %{x:.2%}<br>Annual Return: %{y:.2%}<extra></extra>'
    ))

    fig_ef.update_layout(
        title="Efficient Frontier-like View",
        xaxis_title="Annual Volatility",
        yaxis_title="Annual Return",
        height=500,
        template="plotly_white"
    )

    fig_ef.show()
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Fama-French Regression Analysis

    Analyze quantile portfolio factor exposures and risk-adjusted returns.
    """)
    return

@app.cell
def _(include_ff_regression, marimo, sfr, quantile_df):
    if include_ff_regression.value:
        ff_results = sfr.run_ff_regression(quantile_df)
        marimo.md(f"""
        ### Factor Loadings

        **FF5 Factor Regression Results**

        {ff_results.to_pandas().to_markdown(index=False)}
        """)
    else:
        marimo.md("☑️ Check the box above to run Fama-French 5-factor regression analysis.")
    return


@app.cell
def _(marimo, metrics):
    marimo.md(f"""
    ## Summary Statistics

    {metrics.to_pandas().to_markdown(index=False)}
    """)
    return


if __name__ == "__main__":
    app.run()
