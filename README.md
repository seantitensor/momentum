# SF-Signal

A template project for developing, researching, and backtesting trading signals.

## Project Structure

```
sf-signal/
├── app/
│   ├── research.py               # Analyze signal characteristics (do not edit)
│   ├── production.py             # View backtest performance (do not edit)
│   └── run_backtest.py           # Run the backtest (edit config only)
├── src/
|   ├── signal_research.py        # Explore and develop signal ideas (edit this)
│   └── create_signal.py          # Your signal implementation (edit this)
├── data/
│   ├── signal.parquet            # Output: Your signal
│   └── weights.parquet           # Output: Backtest weights
└── README.md
```

## Workflow

### 1. **Explore Signal Ideas** (`signal_research.py`)
   - Open interactive notebook to explore data and test signal logic
   - Load historical market data
   - Test different signal calculations
   - Visualize signal properties

   ```bash
   marimo edit app/signal_research.py
   ```

### 2. **Implement Signal** (`create_signal.py`)
   - Copy your signal logic from signal_research.py into `create_signal.py`
   - Customize date ranges, data columns, and calculation logic
   - Save signal to `data/signal.parquet`

   ```bash
   uv run python src/create_signal.py
   ```

### 3. **Research Signal** (`research.py`)
   - Analyze quantile portfolios based on your signal
   - Explore signal characteristics before backtesting
   - Adjust quantile levels interactively
   - View performance metrics by quantile

   ```bash
   marimo edit app/research.py
   ```

### 4. **Run Backtest** (`run_backtest.py`)
   - Run MVO-based backtest on your signal
   - Generates optimal portfolio weights
   - Saves results to `data/weights.parquet`

   ```bash
   uv run python app/run_backtest.py
   ```

### 5. **Analyze Performance** (`production.py`)
   - View final portfolio performance
   - Analyze backtest returns, drawdowns, and metrics
   - Inspect portfolio weights and allocations

   ```bash
   marimo edit app/production.py
   ```

## Data Files

All data files are stored in the `data/` directory:

- **`data/signal.parquet`**: Output from `create_signal.py`
  - Columns: `date`, `barrid`, `alpha` (your signal)
  - Format: Parquet (AlphaSchema)

- **`data/weights.parquet`**: Output from backtest
  - Contains: Portfolio weights and performance data
  - Format: Parquet

## Quick Start

First, create the `data/` folder where outputs will be saved:

```bash
mkdir -p data
```

Then follow the workflow:

```bash
# 1. Explore signal ideas
marimo edit app/signal_research.py

# 2. Implement your signal
# Edit src/create_signal.py with your logic
uv run python src/create_signal.py

# 3. Research signal characteristics
marimo edit app/research.py

# 4. Run backtest
uv run python app/run_backtest.py

# 5. View performance
marimo edit app/production.py
```

## Template Files (Do Not Need to Edit)

The following files in the `app/` folder are templates and should not be modified:
- `app/research.py` - Automatically loads and analyzes your signal
- `app/production.py` - Automatically loads and displays backtest results

**All signal customization happens in `src/create_signal.py`.**

## Configuration

Update `run_backtest.py` if needed:
- `byu_email`: Your BYU email for job submission
- `gamma`: Transaction costs or risk aversion parameter
- `constraints`: Add portfolio constraints
- `slurm`: Adjust computational resources

## Next Steps

1. Develop your signal in `app/signal_research.py`
2. Implement finalized logic in `src/create_signal.py`
3. Run the full pipeline to see results
4. Iterate and refine your approach

---

**Note**: This is a template project. Customize `create_signal.py` with your unique signal logic, then use the workflow above to research and backtest your ideas.
