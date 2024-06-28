import wandb

# Initialize wandb
wandb.init(project="Graph covid Predictions", name="experiment_name")

# Log the db.sqlite3 file as an artifact
artifact = wandb.Artifact("db-sqlite3", type="database")
artifact.add_file("from_gpu/db.sqlite3")
wandb.log_artifact(artifact)