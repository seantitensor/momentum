import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo
    import polars as pl
    import plotly.graph_objects as go
    import sf_quant.performance as sfp
    import sf_quant.research as sfr
    import altair as alt
    return alt, go, marimo, pl, sfp, sfr


@app.cell
def _(marimo):
    marimo.md("""
    # Portfolio Performance

    > **Required:** weights directory (`date` · `barrid` · `weight`) + signal file (`date` · `barrid` · `alpha` · `return`)
    """)
    return


@app.cell
def _(marimo):
    data_dir = marimo.ui.text(value="data", label="Weights directory:")
    signal_file = marimo.ui.text(value="data/signal.parquet", label="Signal file:")
    marimo.hstack([data_dir, signal_file])
    return data_dir, signal_file


@app.cell
def _(data_dir, pl):
    import glob
    _files = sorted(glob.glob(f"{data_dir.value}/[0-9][0-9][0-9][0-9].parquet"))
    weights_raw = pl.read_parquet(_files)
    return (weights_raw,)


@app.cell
def _(pl, signal_file):
    signal_raw = pl.read_parquet(signal_file.value)
    return (signal_raw,)


@app.cell
def _(marimo, weights_raw):
    marimo.stop(weights_raw.is_empty(), marimo.md("**⚠️ No year parquet files found in the weights directory**"))
    return


@app.cell
def _(marimo, signal_raw):
    marimo.stop(signal_raw.is_empty(), marimo.md("**⚠️ Signal file not found or empty**"))
    return


@app.cell
def _(marimo):
    import datetime
    sample_cutoff = marimo.ui.date(value=datetime.date(2018, 1, 1), label="Sample cutoff:")
    sample_mode = marimo.ui.radio(
        options=["Full Sample", "In Sample", "Out of Sample"],
        value="Full Sample",
        label="Sample period:",
    )
    marimo.hstack([sample_mode, sample_cutoff])
    return sample_cutoff, sample_mode


@app.cell
def _(pl, sample_cutoff, sample_mode, signal_raw, weights_raw):
    if sample_mode.value == "In Sample":
        weights = weights_raw.filter(pl.col("date") <= sample_cutoff.value)
        signal_df = signal_raw.filter(pl.col("date") <= sample_cutoff.value)
    elif sample_mode.value == "Out of Sample":
        weights = weights_raw.filter(pl.col("date") > sample_cutoff.value)
        signal_df = signal_raw.filter(pl.col("date") > sample_cutoff.value)
    else:
        weights = weights_raw
        signal_df = signal_raw
    return signal_df, weights


@app.cell
def _(marimo, weights):
    _min = weights.select("date").min().item()
    _max = weights.select("date").max().item()
    marimo.md(f"**Date range:** {_min} → {_max} &nbsp;&nbsp; **({weights['date'].n_unique()} trading days)**")
    return


@app.cell
def _(pl, sfp, weights):
    portfolio_returns = (
        sfp.generate_returns_from_weights(weights).with_columns(pl.col('return').truediv(100))
    )
    portfolio_returns
    return (portfolio_returns,)


@app.cell
def _(alt, portfolio_returns):
    _chart = (
        alt.Chart(portfolio_returns)
        .mark_bar()
        .encode(
            x=alt.X("return", type="quantitative", bin=True, title="return"),
            y=alt.Y("count()", type="quantitative", title="Number of records"),
            tooltip=[
                alt.Tooltip(
                    "return",
                    type="quantitative",
                    bin=True,
                    title="return",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "count()",
                    type="quantitative",
                    format=",.0f",
                    title="Number of records",
                ),
            ],
        ).properties(width="container").configure_view(stroke=None)
    )
    _chart
    return


@app.cell
def _():
    return


@app.cell
def _(marimo, portfolio_returns):
    _min_r = portfolio_returns["return"].min()
    _max_r = portfolio_returns["return"].max()
    _nan_count = portfolio_returns["return"].is_nan().sum()
    _inf_count = portfolio_returns["return"].is_infinite().sum()
    marimo.md(f"""
    **Portfolio returns diagnostics:**
    min={_min_r:.6f}, max={_max_r:.6f}, nan={_nan_count}, inf={_inf_count}
    """)
    return


@app.cell
def _(sfp, weights):
    leverage = sfp.generate_leverage_from_weights(weights)
    return (leverage,)


@app.cell
def _(pl, portfolio_returns):
    drawdown = (
        portfolio_returns.sort("date")
        .with_columns(pl.col("return").log1p().cum_sum().alias("_log_val"))
        .with_columns(pl.col("_log_val").cum_max().alias("_log_peak"))
        .with_columns(
            (pl.col("_log_val") - pl.col("_log_peak")).exp().sub(1).alias("drawdown")
        )
        .select("date", "drawdown")
    )
    return (drawdown,)


@app.cell
def _(marimo):
    marimo.md("""
    ## Performance Summary
    """)
    return


@app.cell
def _(marimo, portfolio_returns, sfp):
    _summary = sfp.generate_returns_summary_table(portfolio_returns)
    marimo.md(f"""
    {_summary.to_pandas().to_markdown(index=False)}
    """)
    return


@app.cell
def _(drawdown, marimo, sfp):
    _dd_summary = sfp.generate_drawdown_summary_table(drawdown)
    marimo.md(f"""
    ### Drawdown Summary

    {_dd_summary.to_pandas().to_markdown(index=False)}
    """)
    return


@app.cell
def _(leverage, marimo, sfp):
    _lev_summary = sfp.generate_leverage_summary_table(leverage)
    marimo.md(f"""
    ### Leverage Summary

    {_lev_summary.to_pandas().to_markdown(index=False)}
    """)
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Cumulative Returns
    """)
    return


@app.cell
def _(go, marimo, pl, portfolio_returns):
    _data = (
        portfolio_returns
        .sort("date")
        .with_columns(pl.col("return").log1p().cum_sum().alias("cum_ret"))
    )
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_data.select("date").to_numpy().flatten(),
        y=_data.select("cum_ret").to_numpy().flatten() * 100,
        mode='lines',
        name='Portfolio',
        line=dict(color='steelblue', width=2),
        fill='tozeroy',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Cum Log Return: %{y:.2f}<extra></extra>'
    ))
    _fig.update_layout(
        title="Portfolio Cumulative Log Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Log Return (%)",
        hovermode='x unified',
        height=500,
        template="plotly_white"
    )
    marimo.ui.plotly(_fig)
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Drawdown
    """)
    return


@app.cell
def _(drawdown, go, marimo):
    _fig_dd = go.Figure()
    _fig_dd.add_trace(go.Scatter(
        x=drawdown.select("date").to_numpy().flatten(),
        y=drawdown.select("drawdown").to_numpy().flatten() * 100,
        mode='lines',
        name='Drawdown',
        line=dict(color='crimson', width=2),
        fill='tozeroy',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}<extra></extra>'
    ))
    _fig_dd.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )
    marimo.ui.plotly(_fig_dd)
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Leverage
    """)
    return


@app.cell
def _(go, leverage, marimo):
    _fig_lev = go.Figure()
    _fig_lev.add_trace(go.Scatter(
        x=leverage.select("date").to_numpy().flatten(),
        y=leverage.select("leverage").to_numpy().flatten(),
        mode='lines',
        name='Leverage',
        line=dict(color='steelblue', width=2),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Leverage: %{y:.2f}<extra></extra>'
    ))
    _fig_lev.update_layout(
        title="Portfolio Leverage",
        xaxis_title="Date",
        yaxis_title="Leverage",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )
    marimo.ui.plotly(_fig_lev)
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Turnover
    """)
    return


@app.cell
def _(marimo, sfp, weights):
    _to_stats = sfp.get_turnover_stats(weights)
    marimo.md(f"""
    ### Turnover Summary

    {_to_stats.to_pandas().to_markdown(index=False)}
    """)
    return


@app.cell
def _(pl, weights):
    turnover = (
        weights
        .sort("date", "barrid")
        .with_columns(
            pl.col("weight")
            .sub(pl.col("weight").shift(1))
            .over("barrid")
            .alias("diff")
        )
        .group_by("date")
        .agg(pl.col("diff").abs().sum().alias("two_sided_turnover"))
        .sort("date")
        .with_columns(pl.col("two_sided_turnover").rolling_mean(252))
    )
    return (turnover,)


@app.cell
def _(go, marimo, turnover):
    _fig_to = go.Figure()
    _fig_to.add_trace(go.Scatter(
        x=turnover.select("date").to_numpy().flatten(),
        y=turnover.select("two_sided_turnover").to_numpy().flatten(),
        mode='lines',
        name='Turnover',
        line=dict(color='steelblue', width=2),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Turnover: %{y:.2f}<extra></extra>'
    ))
    _fig_to.update_layout(
        title="Two-Sided Turnover (Rolling 252-Day Mean)",
        xaxis_title="Date",
        yaxis_title="Two-Sided Turnover",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )
    marimo.ui.plotly(_fig_to)
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Monthly Returns Heatmap
    """)
    return


@app.cell
def _(go, marimo, pl, portfolio_returns):
    _MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    _monthly = (
        portfolio_returns
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        ])
        .group_by(["year", "month"])
        .agg((pl.col("return").add(1).product().sub(1).mul(100)).alias("monthly_return"))
        .sort("year", "month")
    )

    _heatmap = (
        _monthly
        .with_columns(pl.col("month").cast(pl.String))
        .pivot(values="monthly_return", index="year", on="month", aggregate_function="first")
        .sort("year")
    )

    _years = _heatmap.select("year").to_numpy().flatten().tolist()
    _z = []
    for _yr_row in _heatmap.iter_rows(named=True):
        _z.append([_yr_row.get(str(m), None) for m in range(1, 13)])

    _fig_hm = go.Figure(go.Heatmap(
        z=_z,
        x=_MONTHS,
        y=[str(y) for y in _years],
        colorscale=[[0.0, "#d73027"], [0.5, "#f7f7f7"], [1.0, "#1a9850"]],
        zmid=0,
        text=[[f"{v:.1f}%" if v is not None else "" for v in _row] for _row in _z],
        texttemplate="%{text}",
        hovertemplate="Month: %{x}<br>Year: %{y}<br>Return: %{z:.2f}%<extra></extra>",
        showscale=True,
    ))
    _fig_hm.update_layout(
        title="Monthly Returns Heatmap (%)",
        height=max(300, 50 * len(_years) + 100),
        template="plotly_white",
    )
    marimo.ui.plotly(_fig_hm)
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Information Coefficient (IC)
    """)
    return


@app.cell
def _(pl, sfp, signal_df):
    ics = sfp.generate_alpha_ics(
        signal_df.select("date", "barrid", "alpha"),
        signal_df.filter(pl.col("return").is_not_null()).select("date", "barrid", "return"),
    )
    return (ics,)


@app.cell
def _(ics, marimo, pl):
    _ic_mean = ics.select(pl.col("ic").mean()).item()
    _ic_ir = ics.select((pl.col("ic").mean() / pl.col("ic").std()).alias("icir")).item()
    marimo.md(f"""
    | Metric | Value |
    |--------|-------|
    | IC (mean) | **{_ic_mean:.4f}** |
    | IC IR | **{_ic_ir:.3f}** |
    """)
    return


@app.cell
def _(go, ics, marimo, pl):
    _data = (
        ics
        .sort("date")
        .with_columns(pl.col("ic").fill_null(0).cum_sum().alias("cumulative_ic"))
    )
    _fig_ic = go.Figure()
    _fig_ic.add_trace(go.Scatter(
        x=_data.select("date").to_numpy().flatten(),
        y=_data.select("cumulative_ic").to_numpy().flatten(),
        mode='lines',
        name='Cumulative IC',
        line=dict(color='steelblue', width=2),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Cum IC: %{y:.3f}<extra></extra>'
    ))
    _fig_ic.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.3)
    _fig_ic.update_layout(
        title="Cumulative Information Coefficient",
        xaxis_title="Date",
        yaxis_title="Cumulative Rank IC",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )
    marimo.ui.plotly(_fig_ic)
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Fama-French Regression
    """)
    return


@app.cell
def _(marimo, portfolio_returns, sfr):
    _ff = sfr.run_ff_regression(portfolio_returns)
    marimo.md(f"""
    {_ff.to_pandas().to_markdown(index=False)}
    """)
    return


if __name__ == "__main__":
    app.run()
