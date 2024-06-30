import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from tensorboard.backend.event_processing import event_accumulator
import glob
import os

# Step 1: Inspect the structure of all relevant tables
def inspect_table_structure(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"PRAGMA table_info({table_name})"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Step 2: Extract data from the Optuna study stored in db.sqlite3
def get_study_data(db_path):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        trials.trial_id,
        trial_values.value AS val_loss,
        CAST(trial_params.param_value AS INTEGER) AS num_params
    FROM
        trials
        JOIN trial_params ON trials.trial_id = trial_params.trial_id
        JOIN trial_values ON trials.trial_id = trial_values.trial_id
    WHERE
        trial_params.param_name = 'h_size' AND
        trials.study_id = (SELECT study_id FROM studies WHERE study_name = "GraphRNN mean + attention nightrun 2")
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

