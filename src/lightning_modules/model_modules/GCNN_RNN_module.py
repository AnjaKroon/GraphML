import os
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_management import Split_graph_data
from src.models.GCNN_RNN.GCNN_RNN import GCNN_RNN
from src.models.GraphRNN.Neighbor_Agregation import Neighbor_Aggregation_Simple, Neighbor_Attention_multiHead
import pytorch_lightning as pl
import torch

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
        
class GCNN_RNNModule(pl.LightningModule):
    def __init__(self, config, fixed_edge_weights=None):
        super(GCNN_RNNModule, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        self.real_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.real_device)
        # Setup the model
        

        self.h_size, self.n_out_features = self.calc_params(config["num_params"])
        self.h_size = self.h_size
        self.n_out_features = self.n_out_features
        
        self.model = GCNN_RNN( input_horizon= config["input_hor"], prediction_horizon= config["pred_hor"],
                              n_nodes= config["n_nodes"], n_features= config["n_features"], 
                              n_out_features= self.n_out_features, h_size= self.h_size, 
                              device= self.real_device, dtype= torch.float32, fixed_edge_weights= fixed_edge_weights,
                              mlp_width= config["mlp_width"])
        
        self.preds = []
        self.targets = []
    def calc_params(self, num_params):
        import math
        h_size = math.ceil(math.sqrt(num_params/4))
        print("Stil need to implement calculation params for GCNN_RNNModule. Now using  dummy values")
        return h_size, 10
    
    def criterion(self, output, target):
        MSE = torch.nn.MSELoss()
        return MSE(output[:, -self.config["pred_hor"]:, :, :self.config["n_features"]], target)
    
    def forward(self, x_in, edge_weights=None):

        return self.model(x_in, pred_hor= self.config["pred_hor"])
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["n_epochs"]//self.config["num_lr_steps"], gamma=self.config["lr_decay"])
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        input_edge_weights, input_node_data, target_edge_weights, target_node_data= batch
        
        
        pred = self(input_node_data)
    
        loss = self.criterion(pred, target_node_data)  
        self.input = input_node_data
        self.pred = pred
        self.target = target_node_data
        self.losses = loss      
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_edge_weights, input_node_data, target_edge_weights, target_node_data = batch
        pred = self(input_node_data)
        
        loss = self.criterion(pred, target_node_data)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
        return loss

    def on_train_end(self) -> None:

        if self.config["plot_predictions"]:
            input_hor = self.config["input_hor"]
            pred_hor = self.config["pred_hor"]
            
            plt.figure()
            num_plot_nodes = 15
            colors = plt.cm.jet(np.linspace(0, 1, num_plot_nodes))
            rand_node_idx = np.random.randint(0, self.pred.shape[2]-1, num_plot_nodes)
            for i,node in enumerate(rand_node_idx):
                plt.plot( self.input[0, :, node, 0].cpu().detach().numpy(), label=f"Node {node} Input", color=colors[i])
                plt.plot( self.pred[0, :, node, 0].cpu().detach().numpy(), label=f"Node {node} pred", color=colors[i], linestyle="--")
                plt.scatter(np.arange(self.config["input_hor"], input_hor+pred_hor),
                            self.target[0, :, node].cpu().detach().numpy(), label=f"Node {node} Target", marker="x",
                            color=colors[i])
                plt.plot(self.pred[0, :, node, 0].cpu().detach().numpy(), 
                        label=f"Node {node} Output", linestyle="--", color=colors[i])
            plt.legend(loc = "upper left")
            plt.title(f"Prediction end train ")
            plt.savefig(f"prediction_plots/Prediction end train model_{np.random.randint(1000)}.png", dpi=500)
            plt.close()