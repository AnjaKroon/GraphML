from src.utils.data_management.Split_graph_data import Split_graph_dataset	
import torch	
if __name__ == '__main__':	
    flow_dataset = "data/mobility_data/daily_county2county_2019_01_01.csv"	
    epi_dataset = "data/data_epi/epidemiology.csv"	
    epi_dates = ["2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16",
                 "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24",
                 "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02",
                 "2020-07-03", "2020-07-04", "2020-07-05", "2020-07-06", "2020-07-07", "2020-07-08", "2020-07-09", "2020-07-10",
                 "2020-07-11", "2020-07-12", "2020-07-13", "2020-07-14", "2020-07-15", "2020-07-16", "2020-07-17", "2020-07-18",
                 "2020-07-19", "2020-07-20", "2020-07-21", "2020-07-22", "2020-07-23", "2020-07-24", "2020-07-25", "2020-07-26",
                 "2020-07-27", "2020-07-28", "2020-07-29", "2020-07-30", "2020-07-31", "2020-08-01", "2020-08-02", "2020-08-03",
                 "2020-08-04", "2020-08-05", "2020-08-06", "2020-08-07", "2020-08-08", "2020-08-09", "2020-08-10", "2020-08-11",
                 "2020-08-12", "2020-08-13", "2020-08-14", "2020-08-15", "2020-08-16", "2020-08-17", "2020-08-18", "2020-08-19",]	

    input_hor = 50	
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
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(10):	
        print(f"Epoch: {i}")	
        j= 0
        for input_edge_weights, input_node_data, target_edge_weights, target_node_data in data_loader:	
            print(f"input_edge_weights: {input_edge_weights.shape}")	
            print(f"input_node_data: {input_node_data.shape}")	
            print(f"target_edge_weights: {target_edge_weights.shape}")	
            print(f"target_node_data: {target_node_data.shape}")	
            print("=====================================")	
            print(f"Sparsity of target_node_data: {target_node_data.eq(0).sum().item() / target_node_data.numel()}")
            plt.figure()
            num_plot_nodes = 10
            colors = plt.cm.jet(np.linspace(0, 1, num_plot_nodes))
            rand_node_idx = np.random.randint(0, input_node_data.shape[2]-1, num_plot_nodes)
            for i,node in enumerate(rand_node_idx):
                plt.plot( input_node_data[0, :, node, 0].cpu().detach().numpy(), label=f"Node {node} Input", color=colors[i])
                plt.scatter(np.arange(input_hor, input_hor+pred_hor),
                            target_node_data[0, :, node].cpu().detach().numpy(), label=f"Node {node} Target", marker="x",
                            color=colors[i])
  
            plt.legend(loc = "upper left")
            plt.savefig(f"test_data_plots/test_data{i}_{j}.png")
            plt.close()
            j = j+1
