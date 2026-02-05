import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def __():
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import date
    from sf_quant.data import load_crsp_monthly
    from sf_quant.schema import AlphaSchema
    # Placeholder for ff_regressions
    # from sf_quant.regressions import ff_regressions
    return pl, np, go, px, date, load_crsp_monthly, AlphaSchema


@app.cell
def __(marimo):
    signal_file = marimo.ui.text(
        value="data/signal.parquet",
        label="Signal file path:"
    )
    return signal_file,


@app.cell
def __(pl, AlphaSchema, signal_file, marimo):
    # Load signal with error handling
    import os
    try:
        if not os.path.exists(signal_file.value):
            marimo.toast(f"Signal file not found at {signal_file.value}", kind="error")
            signal_df = None
        else:
            signal_df = AlphaSchema.read_parquet(signal_file.value)
            marimo.toast(f"Loaded signal with {len(signal_df)} rows", kind="success")
    except Exception as e:
        marimo.toast(f"Error loading signal: {e}", kind="error")
        signal_df = None

    signal_df
    return signal_df,


@app.cell
def __(signal_df, marimo):
    if signal_df is None:
        marimo.stop(marimo.md("**⚠️ Please load a valid signal file first**"))
    return


@app.cell
def __(marimo):
    n_quantiles = marimo.ui.slider(
        value=5,
        start=2,
        stop=10,
        step=1,
        label="Number of quantiles:"
    )
    return n_quantiles,


@app.cell
def __(pl, signal_df, n_quantiles):
    # Create quantile portfolios
    quantile_df = signal_df.with_columns([
        pl.col("alpha")
        .qcut(n_quantiles.value, labels=[f"Q{i+1}" for i in range(n_quantiles.value)])
        .alias("quantile")
    ])

    # Get quantile boundaries for reference
    quantile_boundaries = signal_df.select([
        pl.col("alpha").quantile(pl.col("alpha").count() / (n_quantiles.value)).alias("boundaries")
    ])

    marimo.toast(f"Created {n_quantiles.value} quantile portfolios", kind="success")
    quantile_df.head(10)
    return quantile_df, quantile_boundaries


@app.cell
def __(marimo):
    include_ff_regression = marimo.ui.checkbox(
        value=False,
        label="Include Fama-French regression analysis"
    )
    return include_ff_regression,


@app.cell
def __(pl, quantile_df, load_crsp_monthly):
    # Load returns data
    min_date = quantile_df.select(pl.col("date").min()).item()
    max_date = quantile_df.select(pl.col("date").max()).item()

    returns_df = load_crsp_monthly(
        start=min_date,
        end=max_date,
        columns=["date", "barrid", "ret"]
    )

    # Merge signal with returns
    portfolio_returns = quantile_df.join(
        returns_df,
        on=["date", "barrid"],
        how="inner"
    )

    marimo.toast(f"Loaded returns data with {len(portfolio_returns)} observations", kind="success")
    portfolio_returns.head()
    return portfolio_returns, min_date, max_date, returns_df


@app.cell
def __(pl, portfolio_returns):
    # Calculate quantile portfolio returns (equal-weighted)
    quantile_perf = (
        portfolio_returns
        .group_by(["date", "quantile"])
        .agg(pl.col("ret").mean().alias("portfolio_ret"))
        .sort(["date", "quantile"])
    )

    marimo.toast(f"Calculated quantile portfolio returns", kind="success")
    quantile_perf.head(20)
    return quantile_perf,


@app.cell
def __(pl, quantile_perf):
    # Calculate cumulative returns for each quantile
    cumul_returns = (
        quantile_perf
        .with_columns([
            (1 + pl.col("portfolio_ret")).log().alias("log_ret")
        ])
        .with_columns([
            pl.col("log_ret").cum_sum().over("quantile").exp().alias("cum_ret")
        ])
        .select(["date", "quantile", "cum_ret"])
    )

    cumul_returns
    return cumul_returns,


@app.cell
def __(go, cumul_returns, n_quantiles):
    # Plot cumulative returns
    fig = go.Figure()

    for i in range(n_quantiles.value):
        quantile = f"Q{i+1}"
        data = cumul_returns.filter(pl.col("quantile") == quantile)
        dates = data.select("date").to_numpy().flatten()
        returns = data.select("cum_ret").to_numpy().flatten()

        fig.add_trace(go.Scatter(
            x=dates,
            y=returns,
            mode='lines',
            name=quantile,
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
    return fig,


@app.cell
def __(pl, quantile_perf):
    # Calculate performance metrics
    metrics = (
        quantile_perf
        .with_columns([
            (1 + pl.col("portfolio_ret")).log().alias("log_ret")
        ])
        .group_by("quantile")
        .agg([
            pl.col("portfolio_ret").mean().alias("mean_ret"),
            pl.col("portfolio_ret").std().alias("std_ret"),
            pl.col("log_ret").sum().alias("cum_log_ret"),
            pl.col("portfolio_ret").count().alias("n_obs"),
        ])
        .with_columns([
            (pl.col("cum_log_ret").exp() - 1).alias("total_ret"),
            (pl.col("mean_ret") / pl.col("std_ret")).alias("sharpe_ratio"),
            (pl.col("mean_ret") * 12).alias("annual_ret"),  # Assuming monthly data
            (pl.col("std_ret") * np.sqrt(12)).alias("annual_vol"),
        ])
        .select([
            "quantile",
            "annual_ret",
            "annual_vol",
            "sharpe_ratio",
            "total_ret",
            "n_obs"
        ])
        .sort("quantile")
    )

    marimo.toast(f"Calculated performance metrics", kind="success")
    metrics
    return metrics,


@app.cell
def __(go, metrics):
    # Plot Sharpe Ratios
    fig_sharpe = go.Figure()

    quantiles = metrics.select("quantile").to_numpy().flatten()
    sharpe = metrics.select("sharpe_ratio").to_numpy().flatten()

    fig_sharpe.add_trace(go.Bar(
        x=quantiles,
        y=sharpe,
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
    return fig_sharpe,


@app.cell
def __(go, metrics):
    # Plot Annual Returns vs Volatility
    quantiles = metrics.select("quantile").to_numpy().flatten()
    ret = metrics.select("annual_ret").to_numpy().flatten()
    vol = metrics.select("annual_vol").to_numpy().flatten()

    fig_ef = go.Figure()

    fig_ef.add_trace(go.Scatter(
        x=vol,
        y=ret,
        mode='markers+text',
        marker=dict(size=12, color='steelblue'),
        text=quantiles,
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
    return fig_ef,


@app.cell
def __(marimo, include_ff_regression):
    marimo.md("""
    ## Fama-French Regression Analysis

    The following is a placeholder for Fama-French factor regression analysis.
    """)

    return


@app.cell
def __(marimo, include_ff_regression):
    if include_ff_regression.value:
        marimo.md("""
        ### Factor Loadings

        ⚠️ **Placeholder**: `ff_regressions` from `sf_quant.regressions` would be used here to compute:
        - Market factor loadings
        - Size (SMB) factor loadings
        - Value (HML) factor loadings
        - Momentum (MOM) factor loadings
        - Risk-free rate adjustments

        Example usage:
        ```python
        from sf_quant.regressions import ff_regressions

        results = ff_regressions(
            quantile_portfolio_returns,
            factors_data,
            factor_columns=['MKT', 'SMB', 'HML', 'MOM']
        )
        ```
        """)
    else:
        marimo.md("Click the checkbox above to see Fama-French regression details.")

    return


@app.cell
def __(marimo, metrics):
    marimo.md(f"""
    ## Summary Statistics

    {metrics.to_pandas().to_markdown(index=False)}
    """)
    return


if __name__ == "__main__":
    app.run()
