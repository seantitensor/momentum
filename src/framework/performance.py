import os
from sf_quant.schema import PortfolioRetSchema
from sf_quant.performance import (
    generate_returns_chart,
    generate_drawdown_from_returns,
    generate_drawdown_chart
)

def analyze_performance(results_path: str):
    """
    Analyzes the backtest results.
    """
    print(f"Loading results from {results_path}...")
    # Load returns
    # Note: Ensure the backtest actually produced this file at this path.
    # If using sf-backtester, check the output structure.
    returns = PortfolioRetSchema.read_parquet(results_path)

    # 1. Generate Returns Chart
    print("Generating returns chart...")
    generate_returns_chart(
        returns,
        title="Strategy Performance",
        subtitle="Cumulative Returns",
        file_name="returns_chart.html"
    )

    # 2. Calculate and Chart Drawdowns
    print("Calculating drawdowns...")
    drawdowns = generate_drawdown_from_returns(returns)
    
    print("Generating drawdown chart...")
    generate_drawdown_chart(
        drawdowns,
        title="Strategy Drawdown",
        file_name="drawdown_chart.html"
    )
    
    print("Performance analysis complete.")

if __name__ == "__main__":
    # Assuming the backtest outputs 'returns.parquet' in the 'backtest_results' directory
    # You might need to adjust this path based on actual backtest output structure
    results_file = os.path.join("backtest_results", "returns.parquet")
    
    if os.path.exists(results_file):
        analyze_performance(results_file)
    else:
        print(f"Results file not found at {results_file}. Run the backtest first.")
