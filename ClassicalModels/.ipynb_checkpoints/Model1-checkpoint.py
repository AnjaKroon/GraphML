import torch
import torch.sparse as sparse
from Neighbor_Agregation import Neighbor_Aggregation
from tqdm import tqdm
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoTCN
import pandas as pd

class Temporal_Processing(torch.nn.Module):
    def __init__(self, df, n_nodes, n_features, h_size, f_out_size, fixed_edge_weights=None , device='cpu', dtype=torch.float32):
        """ Initialize the Graph RNN
        Args:
            n_nodes (int): number of nodes in the graph
            n_features (int): number of features in the input tensor
            h_size (int): size of the hidden state
            f_size (int): size of the vector returned by the neighbor aggregation function
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_time_steps, n_edges, 3), per edge (node1 , node2, edge_feat)
            """
        self.internal__init__(dataframe=df)
    
    def internal__init__(self, dataframe):
        #data = prepare_data(dataframe, freq='D', horizon=7)
        """ AutoTCN (h, loss=MAE(), valid_loss=None, config=None,
          search_alg=<ray.tune.search.basic_variant.BasicVariantGenerator
          object at 0x7f3545577df0>, num_samples=10, refit_with_val=False,
          cpus=4, gpus=0, verbose=False, alias=None, backend='ray',
          callbacks=None)
        """
        
        models = [AutoTCN(h=7,
                   num_samples=30,
                   verbose=True)]

        model = NeuralForecast(models=models, freq='D')
        #dataframe = dataframe.rename(columns={"time": "ds", "new_confirmed": "y"})
        print(dataframe)
        #dataframe = pd.melt(dataframe, value_vars="time")#ignore_index=False)
        #dataframe = dataframe.unstack(0)
        print(dataframe.index.get_level_values(0))
        print(dataframe.index.get_level_values(1))
        print(dataframe.index.get_level_values(2))
        print(dataframe.index.get_level_values(3))
        print(dataframe.index)
        

        #print(dataframe.loc[values, columns])
        idx = pd.IndexSlice
        #print(dataframe.loc[:, idx["new_confirmed", "time":"time"]])
        
        
        model.fit(dataframe)
        forecasts = model.predict()