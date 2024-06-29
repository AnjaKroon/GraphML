from tqdm import tqdm

import math
import torch
from torch import nn
import numpy as np

from components.parametric_graph_filter import ParametricGraphFilter
from components.space_time_pooling import SpaceTimeMaxPooling
from torch_geometric.nn import GCNConv



class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, num_params, input_horizon, prediction_horizon, num_features, input_dim, output_dim, num_nodes_kron, num_nodes_pred):
        super(GraphConvolutionalNetwork, self).__init__()
        self.hidden_dim_low, self.hidden_dim_high = get_parameters(num_params)
        self.mygconv1 = GCNConv(input_dim, self.hidden_dim_low)
        self.gconv2 = GCNConv(self.hidden_dim_low, self.hidden_dim_low)
        # same MLP trained per node, just declare it here
        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_dim_low, self.hidden_dim_high), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim_high, output_dim) 
        )
        self.num_features = num_features
        self.num_nodes_kron = num_nodes_kron
        self.num_nodes_pred = num_nodes_pred
        self.features = ['weather1', 'weather2', 'cumulative_confirmed', 'cumulative_deceased', 'new_deceased', 'cumulative_persons_fully_vaccinated', 'new_persons_fully_vaccinated']

    
    def forward(self, data):
        adj_as_edge_index = self.process_input(data)

        x = torch.tensor(self.graph_signal_matrix, dtype=torch.float32)
        x = x.reshape(1, x.shape[0], x.shape[1])
        x = self.mygconv1(x, adj_as_edge_index)
        x = torch.relu(x)
        x = self.gconv2(x, adj_as_edge_index)
        x = torch.relu(x)
        x = x[:, -self.num_nodes_pred:, :] # based on the assumption that the last nodes in the sequence are the most prevalent 
        # in the prediction
        x = x.view(self.num_nodes_pred, self.hidden_dim_low) # unsure if this is working exactly how I want it to
        # x = x.view(x.shape[1]) # change from [1, 3070, 10] to [3070, 10]
        x = self.MLP(x)
        return x
    
    def preprocess(self, flow_dataset, epi_dataset, locations_data, plottable=True):
        preproc = preprocessor_final_data.Preprocessor(flow_dataset, epi_dataset, locations_data, plottable=True)
        graph_kronecker_whole_df = preproc.combined_manual_kronecker() # makes the pandas df from the kronecker data
        num_nodes_per_day = len(list_of_geoids)
        adj_kronecker_whole = get_adj_from_plot(graph_kronecker_whole_df)

        # Convert adjacency to edge_index
        adj_as_edge_index = torch.tensor(adjacency.nonzero(), dtype=torch.long)
    
    def process_input(self, data):
        adjacency, graph_signal, target_graph_signal = data
        
        # Ensure graph_signal should have shape [number_of_nodes_kron, input_dim]
        self.graph_signal_matrix = np.zeros((self.num_nodes_kron, self.num_features)) # the graph signal matrix will change per training example

        for cur_node, element in enumerate(list(graph_signal.values())):
            for i in range(self.num_features):
                self.graph_signal_matrix[cur_node, i] = np.squeeze(element[self.features[i]])

        # needs to have shape [num_nodes_pred, input_dim]
        self.target_graph_signal_matrix = np.zeros((self.num_nodes_pred, self.num_features)) # the graph signal matrix will change per training example

        for cur_node, element in enumerate(list(target_graph_signal.values())):
            for i in range(self.num_features):
                self.target_graph_signal_matrix[cur_node, i] = np.squeeze(element[self.features[i]])
        self.target_graph_signal_matrix = torch.tensor(self.target_graph_signal_matrix, dtype=torch.float32)
        # target_graph_signal = torch.tensor(list(target_graph_signal.values()), dtype=torch.float32).view(-1, output_dim)

        # Convert adjacency to edge_index
        adj_as_edge_index = torch.tensor(adjacency.nonzero(), dtype=torch.long)
        
        return adj_as_edge_index

    def getLoss(self, output):
        return nn.MSELoss()(output, self.target_graph_signal_matrix)

def get_parameters(desired_params):
    mlp_params = round_up_to_nearest_hundred(desired_params*(2/3))
    gcn_params = round_up_to_nearest_hundred(desired_params*(1/3))

    low_params = round_to_nearest_ten(max(solve_quadratic(1, 2, -gcn_params)))
    high_params = round_to_nearest_ten(mlp_params / (low_params + 2))

    return low_params, high_params

def round_up_to_nearest_hundred(number):
    return math.ceil(number / 100) * 100

def round_to_nearest_ten(number):
    return round(number / 10) * 10

def solve_quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None, None  # No real solutions
    else:
        # Calculate the two solutions
        sol1 = (-b + math.sqrt(discriminant)) / (2*a)
        sol2 = (-b - math.sqrt(discriminant)) / (2*a)
        return sol1, sol2