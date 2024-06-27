import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from src.lightning_modules.model_modules.GraphRNN_module import GraphRNNModule
from src.lightning_modules.data_modules.Split_graph_data_module import Split_graph_data_module


def objective(trial):
    config = {
        "n_epochs": 2000,
        "num_train_dates": 75,
        "num_validation_dates": 13,
        "input_hor": 7,
        "pred_hor": 1,
        "h_size": trial.suggest_int("h_size", 64, 128),
        "batch_size": trial.suggest_int("batch_size", 1, 8),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "max_grad_norm": 1,
        "num_lr_steps": 5,
        "lr_decay": 0.5,
        "normalize_edge_weights": trial.suggest_categorical("normalize_edge_weights", [True, False]),
        "neighbor_aggregation": trial.suggest_categorical("neighbor_aggregation", ["mean", "attention"]),
        "n_nodes": 3070,
        "n_features": 1,
        "n_heads": trial.suggest_int("n_heads", 1, 4),
        "profile": False,
    }
    
    flow_dataset = "data/mobility_data/daily_county2county_2019_01_01.csv"
    epi_dataset = "data/data_epi/epidemiology.csv"
    epi_dates = ["2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02", "2020-07-03", "2020-07-04", "2020-07-05", "2020-07-06", "2020-07-07", "2020-07-08", "2020-07-09", "2020-07-10", "2020-07-11", "2020-07-12", "2020-07-13", "2020-07-14", "2020-07-15", "2020-07-16", "2020-07-17", "2020-07-18", "2020-07-19", "2020-07-20", "2020-07-21", "2020-07-22", "2020-07-23", "2020-07-24", "2020-07-25", "2020-07-26", "2020-07-27", "2020-07-28", "2020-07-29", "2020-07-30", "2020-07-31", "2020-08-01", "2020-08-02", "2020-08-03", "2020-08-04", "2020-08-05", "2020-08-06", "2020-08-07", "2020-08-08", "2020-08-09", "2020-08-10", "2020-08-11", "2020-08-12", "2020-08-13", "2020-08-14", "2020-08-15", "2020-08-16", "2020-08-17",
                    "2020-08-18", "2020-08-19", "2020-08-20", "2020-08-21", "2020-08-22", "2020-08-23", "2020-08-24", "2020-08-25", "2020-08-26", "2020-08-27", "2020-08-28", "2020-08-29", "2020-08-30", "2020-08-31", "2020-09-01", "2020-09-02", "2020-09-03", "2020-09-04", "2020-09-05", "2020-09-06", "2020-09-07", "2020-09-08", "2020-09-09", "2020-09-10", "2020-09-11", "2020-09-12", "2020-09-13", "2020-09-14", "2020-09-15", "2020-09-16", "2020-09-17", "2020-09-18", "2020-09-19", "2020-09-20", "2020-09-21", "2020-09-22", "2020-09-23", "2020-09-24", "2020-09-25", "2020-09-26", "2020-09-27", "2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01", "2020-10-02", "2020-10-03", "2020-10-04", "2020-10-05", "2020-10-06", "2020-10-07", "2020-10-08", "2020-10-09", "2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14", "2020-10-15", "2020-10-16", "2020-10-17", "2020-10-18", 
                    "2020-10-19", "2020-10-20", "2020-10-21", "2020-10-22", "2020-10-23", "2020-10-24", "2020-10-25", "2020-10-26", "2020-10-27", "2020-10-28", "2020-10-29", "2020-10-30", "2020-10-31", "2020-11-01", "2020-11-02", "2020-11-03", "2020-11-04", "2020-11-05", "2020-11-06", "2020-11-07", "2020-11-08", "2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12", "2020-11-13", "2020-11-14", "2020-11-15", "2020-11-16", "2020-11-17", "2020-11-18", "2020-11-19", "2020-11-20", "2020-11-21", "2020-11-22", "2020-11-23", "2020-11-24", "2020-11-25", "2020-11-26", "2020-11-27", "2020-11-28", "2020-11-29", "2020-11-30", "2020-12-01", "2020-12-02", "2020-12-03", "2020-12-04", "2020-12-05", "2020-12-06", "2020-12-07", "2020-12-08", "2020-12-09", "2020-12-10", "2020-12-11", "2020-12-12", "2020-12-13", "2020-12-14", "2020-12-15", "2020-12-16", "2020-12-17", "2020-12-18", "2020-12-19",
                    "2020-12-20" ]

    from src.utils.data_management.Split_graph_data import Split_graph_dataset
    data_module = Split_graph_data_module(epi_dates, flow_dataset, epi_dataset, config)
    
    #really ugly fix cus init needs edge weights
    data_set = Split_graph_dataset(
        epi_dates=epi_dates,
        flow_dataset=flow_dataset,
        epi_dataset=epi_dataset,
        input_hor=config["input_hor"],
        pred_hor=config["pred_hor"],
        fake_data=False,
        normalize_edge_weights=config["normalize_edge_weights"]
    )
    print("edge weights shape: ", data_set.edge_weights.shape)
    fixed_edge_weights = data_set.edge_weights[0, :, :]    
    
    model = GraphRNNModule(config, fixed_edge_weights=fixed_edge_weights)
    
    logger = TensorBoardLogger("tb_logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, min_delta= 5e-3,  mode='min', verbose=True)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=config["n_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=1
        
        
    )
    
    trainer.fit(model, datamodule=data_module)
    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="GraphRNN mean + attention",
                                storage="sqlite:///db.sqlite3", load_if_exists=True)
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters: ", study.best_params)
