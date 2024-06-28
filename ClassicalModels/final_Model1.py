import torch
import torch_geometric as pyg
import functorch
from pytorch_tcn import TCN

"""
needs to be imported like this:
    from <path> import <model_name>
Instantiated like this:
    model = <model_name>(num_params, input_horizon, prediction_horizon, other....) #model init
The used like this:
    prediction = model(input_data) #forward pass
"""

class model1_TCN(torch.nn.Module):
    def __init__(self, num_params, input_horizon, prediction_horizon, n_nodes, n_features, device, dtype):
        super(model1_TCN, self).__init__()
        self.device = device
        self.dtype = dtype
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.input_horizon = input_horizon

        # unused
        self.prediction_horizon = prediction_horizon
        
        # calculate
        num_channels = [n_nodes, n_features, 4]
        kernel_size = 1
        
        self.model = TCN(
                num_inputs = n_features,
                num_channels = num_channels,
                kernel_size = kernel_size, #kernel_size: int = 4,
                input_shape = 'NLC', #input_shape: str = 'NCL',
                lookahead = input_horizon, # default: 0
                #dilations: Optional[ ArrayLike ] = None,
                #dilation_reset: Optional[ int ] = None,
                #dropout: float = 0.1,
                #causal: bool = True,
                #use_norm: str = 'weight_norm',
                #activation: str = 'relu',
                #kernel_initializer: str = 'xavier_uniform',
                #use_skip_connections: bool = False,
                #embedding_shapes: Optional[ ArrayLike ] = None,
                #embedding_mode: str = 'add',
                #use_gate: bool = False,
                #output_projection: Optional[ int ] = None,
                #output_activation: Optional[ str ] = None,
            )
        self.model.to(device)
        
    def forward(self, x_in):
        x_in = x_in[0]
        return self.model(x_in)