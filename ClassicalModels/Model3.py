from tqdm import tqdm

import math
import torch
from torch import nn
import numpy as np

from components.parametric_graph_filter import ParametricGraphFilter
from components.space_time_pooling import SpaceTimeMaxPooling
from torch_geometric.nn import GCNConv

from preprocessor_final_data import Preprocessor
from preprocessor_final_data import draw_network
from preprocessor_final_data import get_adj_from_plot
from torch.utils.data import Dataset


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, num_params, input_horizon, prediction_horizon, num_nodes):
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.features = ['weather1', 'weather2', 'cumulative_confirmed', 'cumulative_deceased', 'new_deceased', 'cumulative_persons_fully_vaccinated', 'new_persons_fully_vaccinated']
        self.num_nodes_kron = num_nodes[0]
        self.num_nodes_pred = num_nodes[1]
        
        self.num_features = input_horizon
        self.input_dim = self.num_features  # Number of features per node
        self.output_dim = self.num_features #* prediction_horizon # Number of output features per node
        
        self.hidden_dim_low, self.hidden_dim_high = get_parameters(num_params)
        
        self.mygconv1 = GCNConv(self.input_dim, self.hidden_dim_low)
        self.gconv2 = GCNConv(self.hidden_dim_low, self.hidden_dim_low)
        # same MLP trained per node, just declare it here
        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_dim_low, self.hidden_dim_high), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim_high, self.output_dim)
        )

    
    def forward(self, data):
        adj_as_edge_index = self.process_input(data)
        adj_as_edge_index = torch.squeeze(adj_as_edge_index, dim=0) # remove the batch dimension

        x = torch.tensor(self.graph_signal_matrix, dtype=torch.float32)
        x = x.reshape(1, x.shape[0], x.shape[1])
        x = self.mygconv1(x, adj_as_edge_index)
        x = torch.relu(x)
        x = self.gconv2(x, adj_as_edge_index)
        x = torch.relu(x)
        x = x[:, -self.num_nodes_pred:, :] # based on the assumption that the last nodes in the sequence are the most prevalent in the prediction
        x = x.view(self.num_nodes_pred, self.hidden_dim_low) # unsure if this is working exactly how I want it to
        # x = x.view(x.shape[1]) # change from [1, #nodes, #features] to [#nodes, #features]
        x = self.MLP(x)
        return x
    
    def process_input(self, data):
        adjacency, graph_signal, target_graph_signal = data
        
        # Ensure graph_signal should have shape [number_of_nodes_kron, input_dim]
        self.graph_signal_matrix = np.zeros((self.num_nodes_kron, self.num_features))  # the graph signal matrix will change per training example

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
        adj_as_edge_index = adjacency.clone().detach()
        
        return adj_as_edge_index

    def getLoss(self, output):
        return nn.MSELoss()(output, self.target_graph_signal_matrix)

class KroneckerDataset(Dataset):
    def __getitem__(self, idx):
        return self.all_examples[idx]
    
    def __len__(self):
        return len(self.all_examples)
    
    def __init__(self, preproc, input_hor, pred_hor): #, train_perc = 0.8, test_perc = 0.2):
        # Get the kronecker
        self.graph_kronecker_whole_df = preproc.combined_manual_kronecker() # makes the pandas df from the kronecker data
        adj_kronecker_whole = get_adj_from_plot(self.graph_kronecker_whole_df)
        
        # do it first only with cumulative_confirmed
        tr_epi = preproc.set_timestep_offset_epi_dataset(from_timestep=0).get_epi_dataset() # will get the epi info for the entire dataset, the entire kronecker, then later we index per training example and pred
        
        self.train_graph_sig = {}
        for day in tr_epi:
            entry_count = 0
            for geoid in day.geoid_o:
                if geoid not in self.train_graph_sig: # which should be for every day
                    self.train_graph_sig[geoid] = {
                        'weather1': [],
                        'weather2': [],
                        'cumulative_confirmed': [],
                        'cumulative_deceased': [],
                        'new_deceased': [],
                        'cumulative_persons_fully_vaccinated': [],
                        'new_persons_fully_vaccinated': []
                    }
        
                self.train_graph_sig[geoid]['weather1'].append(day.weather1[entry_count])
                self.train_graph_sig[geoid]['weather2'].append(day.weather2[entry_count])
                self.train_graph_sig[geoid]['cumulative_confirmed'].append(day.cumulative_confirmed[entry_count])
                self.train_graph_sig[geoid]['cumulative_deceased'].append(day.cumulative_deceased[entry_count])
                self.train_graph_sig[geoid]['new_deceased'].append(day.new_deceased[entry_count])
                self.train_graph_sig[geoid]['cumulative_persons_fully_vaccinated'].append(day.cumulative_persons_fully_vaccinated[entry_count])
                self.train_graph_sig[geoid]['new_persons_fully_vaccinated'].append(day.new_persons_fully_vaccinated[entry_count])
                entry_count += 1
                
        # sort train_graph_sig by geoid_o
        self.train_graph_sig = dict(sorted(self.train_graph_sig.items(), key=lambda item: item[0]))
    
        # Combine all values for each feature across all geo_ids
        weather1_values = []
        weather2_values = []
        cumulative_confirmed_values = []
        cumulative_deceased_values = []
        new_deceased_values = []
        cumulative_persons_fully_vaccinated_values = []
        new_persons_fully_vaccinated_values = []
        
        for geoid in self.train_graph_sig:
            weather1_values.extend(self.train_graph_sig[geoid]['weather1'])
            weather2_values.extend(self.train_graph_sig[geoid]['weather2'])
            cumulative_confirmed_values.extend(self.train_graph_sig[geoid]['cumulative_confirmed'])
            cumulative_deceased_values.extend(self.train_graph_sig[geoid]['cumulative_deceased'])
            new_deceased_values.extend(self.train_graph_sig[geoid]['new_deceased'])
            cumulative_persons_fully_vaccinated_values.extend(self.train_graph_sig[geoid]['cumulative_persons_fully_vaccinated'])
            new_persons_fully_vaccinated_values.extend(self.train_graph_sig[geoid]['new_persons_fully_vaccinated'])
        
        # Calculate mean and std for each feature across all geo_ids
        mean_weather1 = np.mean(weather1_values)
        std_weather1 = np.std(weather1_values)
        
        mean_weather2 = np.mean(weather2_values)
        std_weather2 = np.std(weather2_values)
        
        mean_cumulative_confirmed = np.mean(cumulative_confirmed_values)
        std_cumulative_confirmed = np.std(cumulative_confirmed_values)
        
        mean_cumulative_deceased = np.mean(cumulative_deceased_values)
        std_cumulative_deceased = np.std(cumulative_deceased_values)
        
        mean_new_deceased = np.mean(new_deceased_values)
        std_new_deceased = np.std(new_deceased_values)
        
        mean_cumulative_persons_fully_vaccinated = np.mean(cumulative_persons_fully_vaccinated_values)
        std_cumulative_persons_fully_vaccinated = np.std(cumulative_persons_fully_vaccinated_values)
        
        mean_new_persons_fully_vaccinated = np.mean(new_persons_fully_vaccinated_values)
        std_new_persons_fully_vaccinated = np.std(new_persons_fully_vaccinated_values)
        
        # Normalize each feature list in train_graph_sig across all geo_ids
        for geoid in self.train_graph_sig:
            self.train_graph_sig[geoid]['weather1'] = (self.train_graph_sig[geoid]['weather1'] - mean_weather1) / std_weather1 if std_weather1 != 0 else 0
            self.train_graph_sig[geoid]['weather2'] = (self.train_graph_sig[geoid]['weather2'] - mean_weather2) / std_weather2 if std_weather2 != 0 else 0
            self.train_graph_sig[geoid]['cumulative_confirmed'] = (self.train_graph_sig[geoid]['cumulative_confirmed'] - mean_cumulative_confirmed) / std_cumulative_confirmed if std_cumulative_confirmed != 0 else 0
            self.train_graph_sig[geoid]['cumulative_deceased'] = (self.train_graph_sig[geoid]['cumulative_deceased'] - mean_cumulative_deceased) / std_cumulative_deceased if std_cumulative_deceased != 0 else 0
            self.train_graph_sig[geoid]['new_deceased'] = (self.train_graph_sig[geoid]['new_deceased'] - mean_new_deceased) / std_new_deceased if std_new_deceased != 0 else 0
            self.train_graph_sig[geoid]['cumulative_persons_fully_vaccinated'] = (self.train_graph_sig[geoid]['cumulative_persons_fully_vaccinated'] - mean_cumulative_persons_fully_vaccinated) / std_cumulative_persons_fully_vaccinated if std_cumulative_persons_fully_vaccinated != 0 else 0
            self.train_graph_sig[geoid]['new_persons_fully_vaccinated'] = (self.train_graph_sig[geoid]['new_persons_fully_vaccinated'] - mean_new_persons_fully_vaccinated) / std_new_persons_fully_vaccinated if std_new_persons_fully_vaccinated != 0 else 0
    
        self.all_examples = []
        
        list_of_geoids = preproc.flow.iloc[:]['geoid_o'].unique()
        num_nodes_per_day = len(list_of_geoids)

        for example_num in range(0, len(tr_epi) - (input_hor)): # this is how many times you can "shift and have valid data to pull from"
            # now draw out the adjacency matrix per example
            width_of_adj_per_example = num_nodes_per_day*input_hor
            shift = num_nodes_per_day # number of nodes per day
        
            offset = (example_num * shift)
            # this should still work in the new approach
            adj_per_example = adj_kronecker_whole[ offset : offset + width_of_adj_per_example,
                                                   offset : offset + width_of_adj_per_example]
            # now drawing out the train_graph_signal per example
        
            # get the graph signal corresponding to the example nodes
            train_graph_sig_per_example = {k: self.train_graph_sig[k] for k in \
                                           list(self.train_graph_sig)[offset : offset + width_of_adj_per_example]}
        
            # get the graph signal corresponding to the [input_hor : input_hor+pred_hor] set of nodes
            train_graph_sig_per_example_pred = {k: self.train_graph_sig[k] for k in \
                                                list(self.train_graph_sig)[offset + (num_nodes_per_day * input_hor) : offset + num_nodes_per_day*(input_hor + pred_hor)]}

            adj_as_edge_index = torch.tensor(adj_per_example.nonzero(), dtype=torch.long)
            example = [adj_as_edge_index, train_graph_sig_per_example, train_graph_sig_per_example_pred]
            self.all_examples.append(example)
        self.num_nodes = (width_of_adj_per_example, num_nodes_per_day * pred_hor) # (length of adj_per_example, length of train_graph_sig_per_example_pred)
        #self.indices = self.all_examples
        
        # remove those examples from all_training_examples
        #self.all_training_examples = self.all_examples[:int(len(self.all_examples)*train_perc)]
        # take the last 20% of the the training examples and use them as test examples
        #self.all_test_examples = self.all_examples[int(len(self.all_examples)*train_perc):]


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