import os
from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig

def run_backtest():
    project_root = os.getcwd()
    signal_path = os.path.join(project_root, "data", "signal.parquet")
    output_dir = os.path.join(project_root, "data")
    logs_dir = os.path.join(project_root, "logs")

    # Define Slurm Configuration
    slurm_config = SlurmConfig(
        n_cpus=4,
        mem="16G",
        time="01:00:00"
    )

    # Define Backtest Configuration
    config = BacktestConfig(
        signal_name="my_first_signal",
        data_path=signal_path,
        gamma=0.05, # Risk aversion? Assuming 0.05 for now
        project_root=project_root,
        byu_email="user@byu.edu", # Update this
        constraints=[], # Add constraints if needed
        slurm=slurm_config,
        output_dir=output_dir,
        logs_dir=logs_dir
    )

    runner = BacktestRunner(config)
    
    # Run the backtest
    # Note: submit(dry_run=False) will likely submit to Slurm or run locally depending on implementation
    runner.submit(dry_run=False)

if __name__ == "__main__":
    run_backtest()
