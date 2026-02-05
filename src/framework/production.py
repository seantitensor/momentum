import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def __():
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from sf_quant.schema import PortfolioRetSchema
    from sf_quant.performance import (
        generate_returns_chart,
        generate_drawdown_from_returns,
        generate_drawdown_chart
    )
    return pl, np, go, px, PortfolioRetSchema, generate_returns_chart, generate_drawdown_from_returns, generate_drawdown_chart


@app.cell
def __(marimo):
    weights_file = marimo.ui.text(
        value="data/weights.parquet",
        label="MVO weights file path:"
    )
    return weights_file,


@app.cell
def __(pl, weights_file, marimo):
    # Load backtest portfolio results/weights
    import os
    try:
        if not os.path.exists(weights_file.value):
            marimo.toast(f"Weights file not found at {weights_file.value}", kind="error")
            portfolio_data = None
        else:
            portfolio_data = pl.read_parquet(weights_file.value)
            marimo.toast(f"Loaded portfolio data with {len(portfolio_data)} observations", kind="success")
    except Exception as e:
        marimo.toast(f"Error loading portfolio data: {e}", kind="error")
        portfolio_data = None

    portfolio_data
    return portfolio_data,


@app.cell
def __(portfolio_data, marimo):
    if portfolio_data is None:
        marimo.stop(marimo.md("**⚠️ Please load valid backtest results first**"))
    return


@app.cell
def __(pl, portfolio_data):
    # Calculate cumulative returns from portfolio data
    # User: customize this based on your portfolio_data structure
    if portfolio_data is not None:
        try:
            # Assuming portfolio_data has a 'ret' column for returns
            cumul_returns = (
                portfolio_data
                .with_columns([
                    (1 + pl.col("ret")).log().alias("log_ret")
                ])
                .with_columns([
                    pl.col("log_ret").cum_sum().alias("cum_log_ret"),
                    (1 + pl.col("log_ret").cum_sum().exp() - 1).alias("cum_ret")
                ])
                .select(["date", "ret", "cum_ret"])
            )

            marimo.toast(f"Calculated cumulative returns", kind="success")
        except Exception as e:
            marimo.toast(f"Error calculating returns: {e}", kind="error")
            cumul_returns = None
    else:
        cumul_returns = None

    cumul_returns
    return cumul_returns,


@app.cell
def __(go, cumul_returns):
    if cumul_returns is not None:
        # Plot cumulative returns
        dates = cumul_returns.select("date").to_numpy().flatten()
        cum_rets = cumul_returns.select("cum_ret").to_numpy().flatten()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=cum_rets,
            mode='lines',
            name='MVO Portfolio',
            line=dict(color='steelblue', width=2),
            fill='tozeroy',
            hovertemplate='<b>MVO Portfolio</b><br>Date: %{x|%Y-%m-%d}<br>Cum Return: %{y:.2%}<extra></extra>'
        ))

        fig.update_layout(
            title="MVO Portfolio Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            height=500,
            template="plotly_white"
        )

        fig.show()
    return fig,


@app.cell
def __(pl, portfolio_data, np, marimo):
    if portfolio_data is not None:
        try:
            # Calculate performance metrics
            # User: customize this based on your portfolio_data structure
            metrics = (
                portfolio_data
                .with_columns([
                    (1 + pl.col("ret")).log().alias("log_ret")
                ])
                .select([
                    pl.col("ret").mean().alias("mean_ret"),
                    pl.col("ret").std().alias("std_ret"),
                    pl.col("log_ret").sum().alias("cum_log_ret"),
                    pl.col("ret").count().alias("n_obs"),
                    pl.col("ret").min().alias("min_ret"),
                    pl.col("ret").max().alias("max_ret"),
                ])
                .with_columns([
                    (pl.col("cum_log_ret").exp() - 1).alias("total_ret"),
                    (pl.col("mean_ret") / pl.col("std_ret")).alias("sharpe_ratio"),
                    (pl.col("mean_ret") * 12).alias("annual_ret"),  # Assuming monthly data
                    (pl.col("std_ret") * np.sqrt(12)).alias("annual_vol"),
                ])
            )

            marimo.toast(f"Calculated performance metrics", kind="success")
        except Exception as e:
            marimo.toast(f"Error calculating metrics: {e}", kind="error")
            metrics = None
    else:
        metrics = None

    metrics
    return metrics,


@app.cell
def __(marimo, metrics):
    if metrics is not None:
        metrics_df = metrics.to_pandas().iloc[0]
        marimo.md(f"""
        ## Performance Summary

        | Metric | Value |
        |--------|-------|
        | Annual Return | {metrics_df['annual_ret']:.2%} |
        | Annual Volatility | {metrics_df['annual_vol']:.2%} |
        | Sharpe Ratio | {metrics_df['sharpe_ratio']:.3f} |
        | Total Return | {metrics_df['total_ret']:.2%} |
        | Min Monthly Return | {metrics_df['min_ret']:.2%} |
        | Max Monthly Return | {metrics_df['max_ret']:.2%} |
        | Observations | {int(metrics_df['n_obs'])} |
        """)
    return


@app.cell
def __(pl, cumul_returns, go):
    if cumul_returns is not None:
        # Calculate drawdown
        cum_ret_series = cumul_returns.select("cum_ret").to_numpy().flatten()
        running_max = np.maximum.accumulate(cum_ret_series)
        drawdown = (cum_ret_series - running_max) / running_max

        dates = cumul_returns.select("date").to_numpy().flatten()

        fig_dd = go.Figure()

        fig_dd.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='crimson', width=2),
            fill='tozeroy',
            hovertemplate='<b>Drawdown</b><br>Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>'
        ))

        fig_dd.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            hovermode='x unified',
            height=400,
            template="plotly_white"
        )

        fig_dd.show()

        max_drawdown = drawdown.min()
    return fig_dd, max_drawdown


@app.cell
def __(pl, portfolio_data, go):
    if portfolio_data is not None:
        try:
            # Distribution of returns
            # User: customize this based on your portfolio_data structure
            returns = portfolio_data.select("ret").to_numpy().flatten()

            fig_dist = go.Figure()

            fig_dist.add_trace(go.Histogram(
                x=returns,
                nbinsx=30,
                marker_color='steelblue',
                hovertemplate='Return Range: %{x:.2%}<br>Frequency: %{y}<extra></extra>'
            ))

            fig_dist.update_layout(
                title="Distribution of Monthly Returns",
                xaxis_title="Monthly Return",
                yaxis_title="Frequency",
                height=400,
                template="plotly_white"
            )

            fig_dist.show()
        except Exception as e:
            marimo.toast(f"Error creating distribution plot: {e}", kind="error")
            fig_dist = None
    else:
        fig_dist = None
    return fig_dist,


@app.cell
def __(pl, weights_file):
    # Try to load portfolio weights if file exists
    try:
        weights_df = pl.read_parquet(weights_file.value)
        marimo.toast(f"Loaded portfolio weights from {weights_file.value}", kind="success")
    except FileNotFoundError:
        marimo.toast(f"Weights file not found at {weights_file.value}. Skipping weights analysis.", kind="warning")
        weights_df = None
    except Exception as e:
        marimo.toast(f"Error loading weights: {e}", kind="warning")
        weights_df = None

    weights_df
    return weights_df,


@app.cell
def __(marimo, weights_df):
    if weights_df is not None and len(weights_df) > 0:
        marimo.md("## Current MVO Portfolio Weights")
        weights_df.head(20)
    else:
        marimo.md("*Portfolio weights file not available. Ensure the backtest output includes weights.*")
    return


@app.cell
def __(marimo, max_drawdown, metrics):
    if metrics is not None and max_drawdown is not None:
        metrics_df = metrics.to_pandas().iloc[0]
        calmar = metrics_df['annual_ret'] / abs(max_drawdown) if max_drawdown != 0 else 0

        marimo.md(f"""
        ## Additional Metrics

        | Metric | Value |
        |--------|-------|
        | Maximum Drawdown | {max_drawdown:.2%} |
        | Calmar Ratio | {calmar:.3f} |
        """)
    return


@app.cell
def __(marimo):
    marimo.md("""
    ## Notes

    - Returns are assumed to be monthly
    - MVO (Minimum Variance Optimization) weights are loaded from the backtest output
    - Performance calculations are based on actual backtested returns
    - All metrics (Sharpe, returns, volatility) are annualized from monthly data
    """)
    return


if __name__ == "__main__":
    app.run()
