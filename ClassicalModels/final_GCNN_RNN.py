import torch
import torch_geometric as pyg
import functorch


class GCNN(torch.nn.Module):
    def __init__(self, n_nodes, n_features, n_output_features, device, dtype, fixed_edge_weights) -> None:
        super(GCNN, self).__init__()
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_output_features = n_output_features
        self.device = device
        self.dtype = dtype
        self.fixed_edge_weights = fixed_edge_weights
        
        self.graph_convolution = pyg.nn.GCNConv(n_features, n_output_features)
        
    def forward(self, x, edge_idx, edge_weights=None):
        x_out = torch.zeros(x.shape[0], self.n_nodes, self.n_output_features, device= self.device, dtype= self.dtype)
        
        for i in range(x.shape[0]):            
            x_out[i, :, :] = self.graph_convolution(x[i, :, :], edge_idx[i, :, :], edge_weights[i, :])
        return x_out
"""
needs to be imported like this:
    from <path> import <model_name>
Instantiated like this:
    model = <model_name>(num_params, input_horizon, prediction_horizon, other....) #model init
The used like this:
    prediction = model(input_data) #forward pass
"""

class GCNN_RNN(torch.nn.Module):
    def __init__(self, num_params, input_horizon, prediction_horizon, n_nodes, n_features, n_out_features, h_size, device, dtype, fixed_edge_weights=None):
        super(GCNN_RNN, self).__init__()
        self.device = device
        self.dtype = dtype
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.h_size = h_size
        self.n_out_features = n_out_features
        self.input_horizon = input_horizon
        self.fixed_edge_weights = fixed_edge_weights

        #currently not using these
        self.prediction_horizon = prediction_horizon
        self.num_params = num_params
        
        if dtype != torch.float32:
            raise ValueError("Only float32 is supported")
        self.GCNN = GCNN(n_nodes= n_nodes, n_features= n_features, 
                         n_output_features= n_out_features, device= device, 
                         dtype= dtype, fixed_edge_weights= fixed_edge_weights)
        self.GCNN.to(device)
        
        self.RNN = torch.nn.RNN(n_out_features, h_size, device=device)
        self.RNN.to(device)
        
    def forward(self, x_in, edge_weights=None, pred_hor = 1):
        if edge_weights is not None:
            raise ValueError("Only fixed edge weights are supported")
        if self.fixed_edge_weights is None:
            raise ValueError("Fixed edge weights must be provided")
        batch_size = x_in.shape[0]
        
        x_in = x_in.view(-1, self.n_nodes, self.n_features)
        x_in.to(self.device)
        
        fixed_edge_weights = self.fixed_edge_weights.transpose(0, 1)
        
        dup_fixed_edge_weights = fixed_edge_weights.unsqueeze(0).expand(batch_size*self.input_horizon, -1, -1)
        dup_fixed_edge_idx = dup_fixed_edge_weights[:, :2, :].type(torch.int32)
        dup_fixed_edge_weights = dup_fixed_edge_weights[:, 2, :]
        dup_fixed_edge_idx.to(self.device)
        dup_fixed_edge_weights.to(self.device)
        
        # Extracting node IDs and creating a mapping from IDs to indices
        unique_id = dup_fixed_edge_idx[:,:,:].unique()
        map_id = {j.item(): i for i, j in enumerate(unique_id)}

        # Processing edge Tensor: replacing node IDs with corresponding indices
        for i, batch in enumerate(dup_fixed_edge_idx):
            for j, _ in enumerate(batch[0]):
                dup_fixed_edge_idx[i, 0, j] = map_id[dup_fixed_edge_idx[i, 0, j].item()]
                dup_fixed_edge_idx[i, 1, j] = map_id[dup_fixed_edge_idx[i, 1, j].item()]
        
        x = self.GCNN(x_in, edge_idx = dup_fixed_edge_idx, edge_weights= dup_fixed_edge_weights)
        # I reduce the input horizon by 2, because otherwise the size of x doesn't factor out to these four variables
        x = x.view(batch_size, self.input_horizon-2, self.n_nodes, self.n_out_features)
        
        h = torch.randn(1, 3070, 50)
        for i in range(self.input_horizon-2):
            x_in = x[:, i, :, :]
            x_out, h = self.forward_step(x_in, h)
        print(x_out.shape)
        return x_out
    
    def forward_step(self, x_in, h):
        x_out, h = self.RNN(x_in, h)
        return x_out, h