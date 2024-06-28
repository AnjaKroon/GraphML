import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from src.lightning_modules.model_modules.GraphRNN_module import GraphRNNModule
from src.lightning_modules.data_modules.Split_graph_data_module import Split_graph_data_module
import wandb

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run = wandb.init()

        # Set up your configurations
        config_dict = {
            "n_epochs": 2000,
            "num_train_dates": 300,
            "num_validation_dates": 100,
            "input_hor": 7,
            "pred_hor": 1,
            "h_size": config.h_size,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "max_grad_norm": 1,
            "num_lr_steps": 1,
            "lr_decay": 1,
            "normalize_edge_weights": config.normalize_edge_weights,
            "neighbor_aggregation": config.neighbor_aggregation,
            "n_nodes": 3070,
            "n_features": 1,
            "n_heads": config.n_heads,
            "profile": False,
            "min_delta": 0,
        }
        
        flow_dataset = "data/mobility_data/daily_county2county_2019_01_01.csv"
        epi_dataset = "data/data_epi/epidemiology.csv"
        epi_dates = [
            # (list of dates)
        ]
        
        from src.utils.data_management.Split_graph_data import Split_graph_dataset
        data_module = Split_graph_data_module(epi_dates, flow_dataset, epi_dataset, config_dict)
        
        data_set = Split_graph_dataset(
            epi_dates=epi_dates,
            flow_dataset=flow_dataset,
            epi_dataset=epi_dataset,
            input_hor=config_dict["input_hor"],
            pred_hor=config_dict["pred_hor"],
            fake_data=False,
            normalize_edge_weights=config_dict["normalize_edge_weights"]
        )
        print("edge weights shape: ", data_set.edge_weights.shape)
        fixed_edge_weights = data_set.edge_weights[0, :, :]    
        
        model = GraphRNNModule(config_dict, fixed_edge_weights=fixed_edge_weights)
        
        wandb_logger = WandbLogger()
        checkpoint_callback = ModelCheckpoint(monitor="val_loss")
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, min_delta=config_dict["min_delta"], mode='min', verbose=True)

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            max_epochs=config_dict["n_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            log_every_n_steps=1
        )
        
        trainer.fit(model, datamodule=data_module)
        val_loss = trainer.callback_metrics["val_loss"].item()

        # Log the validation loss to wandb
        wandb.log({"val_loss": val_loss})
        run.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep('sweep_config.yaml', project='Covid prediction Graph', entity='Init entity')
    wandb.agent(sweep_id, train, count=100)
