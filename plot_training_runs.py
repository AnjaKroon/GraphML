
from src.utils.analysis.analyse_training_runs import inspect_table_structure, get_study_data, get_tensorboard_data, plot_results
# Paths to the files
db_path = "from_gpu/db.sqlite3"
tb_log_dir = "tb_logs"

# Inspect table structure
trials_structure = inspect_table_structure(db_path, "trials")
trial_params_structure = inspect_table_structure(db_path, "trial_params")
trial_values_structure = inspect_table_structure(db_path, "trial_values")

print("Trials table structure:\n", trials_structure)
print("Trial params table structure:\n", trial_params_structure)
print("Trial values table structure:\n", trial_values_structure)

# If the structure shows the correct column names, use them in get_study_data function

# Get data
optuna_data = get_study_data(db_path)

# Print extracted data to verify
print("Optuna Data:\n", optuna_data.head())


# Plot results
plot_results(optuna_data)