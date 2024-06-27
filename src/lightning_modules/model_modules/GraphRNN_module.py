import os
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_management import Split_graph_data
from src.models.GraphRNN.GraphRNN import Graph_RNN
from src.models.GraphRNN.Neighbor_Agregation import Neighbor_Aggregation
import pytorch_lightning as pl
import torch

from sklearn.model_selection import train_test_split

class GraphRNNModule(pl.LightningModule):
    def __init__(self, config):
        super(GraphRNNModule, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        self.real_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.real_device)
        # Setup the model
        if config["neighbor_aggregation"] == "mean":
            self.neighbor_aggregator = Neighbor_Aggregation(
                n_nodes=self.config["n_nodes"], h_size=self.config["h_size"],
                f_out_size=self.config["h_size"], fixed_edge_weights=None,  # Adjusted to None for now
                device=self.real_device, dtype=torch.float32
            )
        elif config["neighbor_aggregation"] == "attention":
            self.neighbor_aggregator= None
        else:
            raise NotImplementedError(f"Neighbor aggregation method {config['neighbor_aggregation']} not implemented.")

        self.model = Graph_RNN(
            n_nodes=self.config["n_nodes"], n_features=self.config["n_features"],
            h_size=self.config["h_size"], f_out_size=self.config["h_size"],
            input_hor=self.config["input_hor"], fixed_edge_weights=None,  # Adjusted to None for now
            device=self.real_device, dtype=torch.float32, neighbor_aggregator=self.neighbor_aggregator
        )
    def criterion(self, output, target):
        MSE = torch.nn.MSELoss()
        return MSE(output[:, -self.config["pred_hor"]:, :, :], target)
    
    def forward(self, x_in, edge_weights=None):
        self.model.set_fixed_edge_weights(edge_weights[0,0,:,:])
        return self.model(x_in, pred_hor= self.config["pred_hor"])
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["n_epochs"]//self.config["num_lr_steps"], gamma=self.config["lr_decay"])
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        input_edge_weights, input_node_data, target_edge_weights, target_node_data= batch
        
        output = self(input_node_data, edge_weights=input_edge_weights)
        
        loss = self.criterion(output, target_node_data)        
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_edge_weights, input_node_data, target_edge_weights, target_node_data = batch
        output = self(input_node_data, input_edge_weights)
        
        loss = self.criterion(output, target_node_data)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
