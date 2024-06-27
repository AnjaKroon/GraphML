
from src.utils.data_management.Split_graph_data import Split_graph_dataset
import torch
if __name__ == '__main__':
    flow_dataset = "data/mobility_data/daily_county2county_2019_01_01.csv"
    epi_dataset = "data/data_epi/epidemiology.csv"
    epi_dates = ["2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"]

    input_hor = 4
    pred_hor = 2
    
    data_set = Split_graph_dataset(epi_dates=epi_dates, 
                                flow_dataset=flow_dataset,
                                epi_dataset=epi_dataset,
                                input_hor=input_hor,
                                pred_hor=pred_hor,
                                fake_data=False)

    # data_sampler = GraphRNN_DataSampler(data_set, input_hor=input_hor, pred_hor=pred_hor)
    # data_loader = torch.utils.data.DataLoader(data_set, batch_size=3, sampler=data_sampler, num_workers=3)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=2, num_workers=0)
    
    num_nans = 0
    for input_edge_weights, input_node_data, target_edge_weights, target_node_data in data_loader:
        for batch in range(input_node_data.shape[0]):
            for time in range(input_node_data.shape[1]):
                for node in range(input_node_data.shape[2]):
                    if torch.isnan(input_node_data[batch, time, node, 0]).item():
                        num_nans += 1

    print(f"Number of NaNs in the dataset: {num_nans}")
    
    for i in range(10):
        print(f"Epoch: {i}")
        for input_edge_weights, input_node_data, target_edge_weights, target_node_data in data_loader:
            print(f"input_edge_weights: {input_edge_weights.shape}")
            print(f"input_node_data: {input_node_data.shape}")
            print(f"target_edge_weights: {target_edge_weights.shape}")
            print(f"target_node_data: {target_node_data.shape}")
            print("=====================================")
            print(f"Sparsity of target_node_data: {target_node_data.eq(0).sum().item() / target_node_data.numel()}")
        
       
