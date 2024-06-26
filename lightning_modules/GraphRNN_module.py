import os
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GraphRNN.GraphRNN_utils import GraphRNN_dataset
from GraphRNN.GraphRNN import Graph_RNN, Neighbor_Aggregation
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import torch
import pandas as pd
from tqdm import tqdm
from GraphRNN.GraphRNN_utils import GraphRNN_dataset
from GraphRNN.GCNN_RNN import Graph_RNN, Neighbor_Aggregation
import matplotlib.pyplot as plt
import json
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

class GraphRNNModule(pl.LightningModule):
    def __init__(self, config):
        super(GraphRNNModule, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.criterion = torch.nn.MSELoss()
        
        # Setup the model
        self.neighbor_aggregator = Neighbor_Aggregation(
            n_nodes=self.config["n_nodes"], h_size=self.config["h_size"],
            f_out_size=self.config["h_size"], fixed_edge_weights=None,  # Adjusted to None for now
            device=self.device, dtype=torch.float32
        )
        
        self.model = Graph_RNN(
            n_nodes=self.config["n_nodes"], n_features=self.config["n_features"],
            h_size=self.config["h_size"], f_out_size=self.config["h_size"],
            input_hor=self.config["input_hor"], fixed_edge_weights=None,  # Adjusted to None for now
            device=self.device, dtype=torch.float32, neighbor_aggregator=self.neighbor_aggregator
        )
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["n_epochs"]//self.config["num_lr_steps"], gamma=self.config["lr_decay"])
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        input_node_data, output, target_node_data = batch
        loss = self.criterion(output[:, -self.config["pred_hor"]:, :, :], target_node_data[:, :self.config["pred_hor"], :, :])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_node_data, output, target_node_data = batch
        loss = self.criterion(output[:, -self.config["pred_hor"]:, :, :], target_node_data[:, :self.config["pred_hor"], :, :])
        self.log('val_loss', loss)
        return loss
