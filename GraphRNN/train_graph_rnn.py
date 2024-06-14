import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import pandas as pd
from tqdm import tqdm
from GraphRNN_utils import GraphRNN_dataset, GraphRNN_DataSampler
from GraphRNN import Graph_RNN, Neighbor_Aggregation
import matplotlib.pyplot as plt

import torch.profiler

def train(model, data_loader, criterion, optimizer, pred_hor, device, n_epochs =10, save=None):
    losses = []
    parameter_mag = {"init_H" : [], "A": [], "B": [], "C": [], "D": [], "E": []}
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch}")
        print("Loading Data")
        epoch_loss = 0
        for input_edge_weights, input_node_data, target_edge_weights, target_node_data in tqdm(data_loader):
            
            for param_name, param in model.named_parameters():
                parameter_mag[param_name].append(param.abs().mean().item())
            
            input_edge_weights = input_edge_weights.to(device)
            input_node_data = input_node_data.to(device)
            target_edge_weights = target_edge_weights.to(device)
            target_node_data = target_node_data.to(device)
            output = model(x_in=input_node_data, edge_weights = input_edge_weights, pred_hor = pred_hor)

            print("output", output[0, -pred_hor:, :10, 0])
            print("target", target_node_data[0, :, :10, 0])
            
            # print(f"output: {output}")
            # print(f"target_node_data: {target_node_data}")
              
            loss = criterion(output[:,-pred_hor:,:,:], target_node_data[:,:pred_hor,:,:])
            loss += 0.1 * criterion(output[:,:input_hor,:,:], input_node_data[:,:,:])
            print(f"batch loss: {loss/((pred_hor+0.1*input_hor) * target_node_data.shape[2] ):.3e}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss)
        print(f"epoch loss: {epoch_loss/((pred_hor+0.1*input_hor) * target_node_data.shape[2] ):.3e}")
    if save is not None:
        torch.save(model.state_dict(), save)
    return losses, parameter_mag





if __name__ == "__main__":
    print("Starting training run...")
    flow_dataset = "data/daily_county2county_2019_01_01.csv"
    epi_dataset = "data_epi/epidemiology.csv"
    epi_dates = ["2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12",
                 "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16",
                 "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20",
                 "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24",
                    "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28",
                    "2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02",
                    "2020-07-03", "2020-07-04", "2020-07-05", "2020-07-06",
                    "2020-07-07", "2020-07-08", "2020-07-09", "2020-07-10",
                    "2020-07-11", "2020-07-12", "2020-07-13", "2020-07-14",
                    "2020-07-15", "2020-07-16", "2020-07-17", "2020-07-18",
                    "2020-07-19", "2020-07-20", "2020-07-21", "2020-07-22",
                    "2020-07-23", "2020-07-24", "2020-07-25", "2020-07-26",
                    "2020-07-27", "2020-07-28", "2020-07-29", "2020-07-30"
                 ]


    input_hor = 6
    pred_hor = 2
    print("Loading data...")
    data_set = GraphRNN_dataset(epi_dates = epi_dates,
                                flow_dataset = flow_dataset,
                                epi_dataset = epi_dataset,
                                input_hor=input_hor,
                                pred_hor=pred_hor,
                                fake_data=False)
    # data_set.visualize(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # data_sampler = GraphRNN_DataSampler(data_set, input_hor=input_hor, pred_hor=pred_hor)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, num_workers=3, shuffle=True)
    
    print("Data loaded.")
    model  = Graph_RNN(n_nodes = data_set.n_nodes,
                       n_features = data_set.n_features,
                       h_size = 30,
                       f_out_size =30,
                       device=device)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    torch.autograd.set_detect_anomaly(False)

    model.to(device)
    
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Use torch.profiler to profile the model
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=0),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        print("Starting training with profiling...")
        losses, parameter_mag = train(model, data_loader,
                                  criterion, optimizer,
                                  pred_hor, device, n_epochs=50,
                                  save="model_state_dict.pth")
        print("Finished training with profiling.")

    # Verify that the log directory is populated
    if os.listdir(log_dir):
        print(f"Log files generated in {log_dir}")
    else:
        print(f"No log files found in {log_dir}")
    

    
    torch.save(model.state_dict(), 'model_state_dict.pth')
        
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()
    
    plt.figure()
    plt.plot(parameter_mag["init_H"], label="init_H")
    plt.plot(parameter_mag["A"], label="A")
    plt.plot(parameter_mag["B"], label="B")
    plt.plot(parameter_mag["C"], label="C")
    plt.plot(parameter_mag["D"], label="D")
    plt.plot(parameter_mag["E"], label="E")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Magnitude")
    plt.yscale("log")
    plt.legend()
    plt.show()
    
