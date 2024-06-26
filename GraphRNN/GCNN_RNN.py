import torch
import torch_geometric as pyg
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
        x = self.graph_convolution(x, edge_index=edge_idx, edge_weight= edge_weights)
        return x    

class GCNN_RNN(torch.nn.Module):
    def __init__(self, n_nodes, n_features, n_out_features, h_size, input_hor, device, dtype, fixed_edge_weights=None):
        super(GCNN_RNN, self).__init__()
        self.device = device
        self.dtype = dtype
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.h_size = h_size
        self.n_out_features = n_out_features
        self.input_hor = input_hor
        self.fixed_edge_weights = fixed_edge_weights
        
        if dtype != torch.float32:
            raise ValueError("Only float32 is supported")
        self.GCNN = GCNN(n_nodes= n_nodes, n_features= n_features, 
                         n_output_features= n_out_features, device= device, 
                         dtype= dtype, fixed_edge_weights= fixed_edge_weights)
        
        self.RNN = torch.nn.RNN(n_out_features, h_size)
        
    def forward(self, x_in, edge_weights=None, pred_hor = 1):
        if edge_weights is not None:
            raise ValueError("Only fixed edge weights are supported")
        if self.fixed_edge_weights is None:
            raise ValueError("Fixed edge weights must be provided")
        batch_size = x_in.shape[0]
        
        x_in = x_in.view(-1, self.n_nodes, self.n_features)
        
        fixed_edge_weights = self.fixed_edge_weights.transpose(0, 1)
        
        dup_fixed_edge_weights = fixed_edge_weights.unsqueeze(0).expand(batch_size*self.input_hor, -1, -1)
        print("dup_fixed_edge_weights.shape)" + str(dup_fixed_edge_weights.shape))
        dup_fixed_edge_idx = dup_fixed_edge_weights[:, :2, :]
        dup_fixed_edge_weights = dup_fixed_edge_weights[:, 2, :]
        print("dup_fixed_edge_idx.shape)" + str(dup_fixed_edge_idx.shape))
        print("dup_fixed_edge_weights.shape)" + str(dup_fixed_edge_weights.shape))
        

        
        x = self.GCNN(x_in, edge_idx = dup_fixed_edge_idx, edge_weights= dup_fixed_edge_weights)
        x = x.view(batch_size, self.input_hor, self.n_nodes, self.n_out_features)
        
        for i in range(self.input_hor):
            x_in = x[:, i, :, :]
            x_out, h = self.forward_step(x_in, h)
        print(x_out.shape)
        return x_out
    
    def forward_step(self, x_in, h):
        x_out, h = self.RNN(x_in, h)
        return x_out, h