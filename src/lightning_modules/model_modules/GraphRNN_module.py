import os
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_management import Split_graph_data
from src.models.GraphRNN.GraphRNN import Graph_RNN
from src.models.GraphRNN.Neighbor_Agregation import Neighbor_Aggregation_Simple, Neighbor_Attention_multiHead
import pytorch_lightning as pl
import torch

from sklearn.model_selection import train_test_split

class GraphRNNModule(pl.LightningModule):
    def __init__(self, config, fixed_edge_weights):
        super(GraphRNNModule, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        self.real_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.real_device)
        self.h_size = self.calc_h_size(config["num_params"])
        # Setup the model
        if config["neighbor_aggregation"] == "mean":
            self.neighbor_aggregator = Neighbor_Aggregation_Simple(
                n_nodes=self.config["n_nodes"], h_size=self.h_size,
                f_out_size=self.h_size, fixed_edge_weights=fixed_edge_weights,  # Adjusted to None for now
                device=self.real_device, dtype=torch.float32
            )
        elif config["neighbor_aggregation"] == "attention":
            self.neighbor_aggregator= Neighbor_Attention_multiHead(
            n_head= config["n_heads"], n_nodes=self.config["n_nodes"], h_size=self.h_size,
            f_out_size=self.h_size, edge_weights=fixed_edge_weights, device=self.device, dtype=torch.float32) # Adjusted to None for now
        elif config["neighbor_aggregation"] == "none":
            self.neighbor_aggregator = None
        else:
            raise NotImplementedError(f"Neighbor aggregation method {config['neighbor_aggregation']} not implemented.")

        self.model = Graph_RNN(
            n_nodes=self.config["n_nodes"], n_features=self.config["n_features"],
            h_size=self.h_size, f_out_size=self.h_size,
            input_hor=self.config["input_hor"], fixed_edge_weights=fixed_edge_weights,  # Adjusted to None for now
            device=self.real_device, dtype=torch.float32, neighbor_aggregator=self.neighbor_aggregator,
            mlp_width=self.config["mlp_width"]
        )
    def calc_h_size(self, num_params):
        # num_params = 2 * h_size**2 + h_size * n_features 
        # + h_size + h_size * h_size + n_features * h_size + n_features**2
        # + 2* (h_size*mlp_width)**2
        # approx= (2*mlp_width**2+ 2 + use_neighbor)*h_size**2
        import math
        if self.config["neighbor_aggregation"] == "mean":
            h_size = math.sqrt(num_params/(2*self.config["mlp_width"]**2 + self.config["mlp_width"]  + 2 + 1))
        elif self.config["neighbor_aggregation"] == "attention":
            h_size = math.sqrt(num_params/(2*self.config["mlp_width"]**2 + self.config["mlp_width"] +  2 + self.config["n_heads"]*2 + 1 ))
        elif self.config["neighbor_aggregation"] == "none":
            h_size = math.sqrt(num_params/(2*self.config["mlp_width"]**2+ self.config["mlp_width"] + 2))
        else:
            raise NotImplementedError(f"Neighbor aggregation method {self.config['neighbor_aggregation']} not implemented.")
        print(f"Calculated h_size: {int(h_size)}")
        if h_size < 1:
            print("h_size is less than 1. Setting it to 1")
            h_size = 1
        self.log("h_size", int(h_size))
        return int(h_size)
      
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
