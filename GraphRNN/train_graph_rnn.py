import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import pandas as pd
from tqdm import tqdm
from GraphRNN_utils import GraphRNN_dataset
from GraphRNN import Graph_RNN, Graph_RNN_init
import matplotlib.pyplot as plt
import json
import torch.profiler
import numpy as np
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42) 

def train(model, train_loader, val_loader, criterion, optimizer, pred_hor, device,
          learning_rate_scheduler, n_epochs =10, save_name=None, max_grad_norm=1, n_plots=None):
    train_losses = []
    val_losses = []
    parameter_mag = {param_name: [] for param_name, param in model.named_parameters()}
    gradients = {}
    gradients["pre_limit"] = {param_name: [] for param_name, param in model.named_parameters()}
    gradients["post_limit"] = {param_name: [] for param_name, param in model.named_parameters()}
    hidden_states = []
    input_hor = model.input_hor

    for epoch in tqdm(range(n_epochs), desc="Epoch"):
        save_string = f"{save_name}_epoch_{epoch}_lr_{optimizer.param_groups[0]['lr']}_batch_{train_loader.batch_size} _pred_{pred_hor}_input_{input_hor}_num_dates_{len(train_loader)}_h_size_{model.h_size}"
        train_loss = 0
        batch_num = 0
        for input_edge_weights, input_node_data, target_edge_weights, target_node_data in train_loader:
            input_hor = input_node_data.shape[1]
            if batch_num==1 and epoch == 1:
                # prof.step()
                pass

            
            input_edge_weights = input_edge_weights.to(device)
            input_node_data = input_node_data.to(device)
            target_edge_weights = target_edge_weights.to(device)
            target_node_data = target_node_data.to(device)
            # output = model(x_in=input_node_data, edge_weights = input_edge_weights, pred_hor = pred_hor)
            output = model(x_in=input_node_data, pred_hor = pred_hor)

            
            # print(f"output: {output}")
            # print(f"target_node_data: {target_node_data}")
              
            loss = criterion(input_node_data, output, target_node_data)
            
            optimizer.zero_grad()
            loss.backward()
            for param_name, param in model.named_parameters():
                try:
                    parameter_mag[param_name].append(param.abs().mean().item())
                    gradients["pre_limit"][param_name].append(param.grad.norm().item())
                except Exception as e:
                    parameter_mag[param_name].append(param.abs().mean().item())
                    gradients["pre_limit"][param_name].append(0)
                     
            hidden_state_mag = model.H.abs().mean().item()
            hidden_states.append(hidden_state_mag)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            for param_name, param in model.named_parameters():
                try:
                    gradients["post_limit"][param_name].append(param.grad.norm().item())
                except:
                    gradients["post_limit"][param_name].append(0)

            optimizer.step()
            train_loss += loss.item()/(len(train_loader))
            batch_num += 1
        
        if epoch % (n_epochs/10) == 0:
            val_loss = 0
            for input_edge_weights, input_node_data, target_edge_weights, target_node_data in val_loader:
                
                input_edge_weights = input_edge_weights.to(device)
                input_node_data = input_node_data.to(device)
                target_edge_weights = target_edge_weights.to(device)
                target_node_data = target_node_data.to(device)
                output = model(x_in=input_node_data, pred_hor = pred_hor)
                
                val_loss += criterion(input_node_data, output, target_node_data)/(len(val_loader))

            print(f"EPOCH: {epoch} ", end="")
            print(f"$ Train Loss: { train_loss:.3e} ", end="")
            print(f"$ Validation Loss: { val_loss} ")
        #Calculate the average loss per prediction, so per node, per time step
 
        train_losses.append(train_loss)
        val_losses.append(val_loss.detach().cpu().tolist())
        if n_plots is not None:
            if epoch % (int(n_epochs)/n_plots)== 0 or epoch == n_epochs-1:
                print(f"$ Output: {output[0, -pred_hor:, :5, 0]}")
                print(f"$ Target: {target_node_data[0, :, :5, 0]}")


                plt.figure()
                num_plot_nodes = 5
                colors = plt.cm.jet(np.linspace(0, 1, num_plot_nodes))
                rand_node_idx = np.random.randint(0, input_node_data.shape[2]-1, num_plot_nodes)
                for i,node in enumerate(rand_node_idx):

                    plt.plot( input_node_data[0, :, node, 0].cpu().detach().numpy(), label=f"Node {node} Input", color=colors[i])
                    plt.scatter(np.arange(input_hor, input_hor+pred_hor),
                                target_node_data[0, :, node, 0].cpu().detach().numpy(), label=f"Node {node} Target", marker="x",
                                color=colors[i])
                    plt.plot(output[0, :, node, 0].cpu().detach().numpy(), 
                            label=f"Node {node} Output", linestyle="--", color=colors[i])
                plt.legend(loc = "upper left")
                plt.title(f"Loss{train_loss}: save_string ")
                plt.savefig(f"GraphML/GraphRNN/plots/covid_predictions/plot_{save_string}_{epoch}.png")
                plt.close()
            
        learning_rate_scheduler.step()
            
    if save_name is not None:
        torch.save(model.state_dict(), f"GraphML/GraphRNN/models/model_state_dict_{save_string}.pth")
        with open(f"GraphML/GraphRNN/losses/losses_6heads_{save_string}.json", "w") as f:
            json.dump(train_losses, f)
        with open(f"GraphML/GraphRNN/losses/ val_6heads_losses_{save_string}.json", "w") as f:
            json.dump(val_losses, f)
    return train_losses, val_losses, parameter_mag, gradients, hidden_states

config = {  "n_epochs": 1,
            "num_train_dates": 60,
            "num_validation_dates": 12,
            "input_hor": 7,
            "pred_hor": 1,
            "h_size": 70,
            "batch_size": 10,
            "lr": 0.01,
            "max_grad_norm": 1,
            "num_lr_steps": 5,
            "lr_decay": 0.5,
            "normalize_edge_weights": False
            
         }



if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    assert config["input_hor"] + config["pred_hor"] <= config["num_train_dates"], "Input and prediction horizon too large for number of training dates"
    assert config["input_hor"] + config["pred_hor"] <= config["num_validation_dates"], "Input and prediction horizon too large for number of validation dates"
    
    
    print("Starting training run...")
    flow_dataset = "GraphML/data/daily_county2county_2019_01_01.csv"
    epi_dataset = "GraphML/data_epi/epidemiology.csv"
    epi_dates = ["2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02", "2020-07-03", "2020-07-04", "2020-07-05", "2020-07-06", "2020-07-07", "2020-07-08", "2020-07-09", "2020-07-10", "2020-07-11", "2020-07-12", "2020-07-13", "2020-07-14", "2020-07-15", "2020-07-16", "2020-07-17", "2020-07-18", "2020-07-19", "2020-07-20", "2020-07-21", "2020-07-22", "2020-07-23", "2020-07-24", "2020-07-25", "2020-07-26", "2020-07-27", "2020-07-28", "2020-07-29", "2020-07-30", "2020-07-31", "2020-08-01", "2020-08-02", "2020-08-03", "2020-08-04", "2020-08-05", "2020-08-06", "2020-08-07", "2020-08-08", "2020-08-09", "2020-08-10", "2020-08-11", "2020-08-12", "2020-08-13", "2020-08-14", "2020-08-15", "2020-08-16", "2020-08-17",
                    "2020-08-18", "2020-08-19", "2020-08-20", "2020-08-21", "2020-08-22", "2020-08-23", "2020-08-24", "2020-08-25", "2020-08-26", "2020-08-27", "2020-08-28", "2020-08-29", "2020-08-30", "2020-08-31", "2020-09-01", "2020-09-02", "2020-09-03", "2020-09-04", "2020-09-05", "2020-09-06", "2020-09-07", "2020-09-08", "2020-09-09", "2020-09-10", "2020-09-11", "2020-09-12", "2020-09-13", "2020-09-14", "2020-09-15", "2020-09-16", "2020-09-17", "2020-09-18", "2020-09-19", "2020-09-20", "2020-09-21", "2020-09-22", "2020-09-23", "2020-09-24", "2020-09-25", "2020-09-26", "2020-09-27", "2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01", "2020-10-02", "2020-10-03", "2020-10-04", "2020-10-05", "2020-10-06", "2020-10-07", "2020-10-08", "2020-10-09", "2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14", "2020-10-15", "2020-10-16", "2020-10-17", "2020-10-18", 
                    "2020-10-19", "2020-10-20", "2020-10-21", "2020-10-22", "2020-10-23", "2020-10-24", "2020-10-25", "2020-10-26", "2020-10-27", "2020-10-28", "2020-10-29", "2020-10-30", "2020-10-31", "2020-11-01", "2020-11-02", "2020-11-03", "2020-11-04", "2020-11-05", "2020-11-06", "2020-11-07", "2020-11-08", "2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12", "2020-11-13", "2020-11-14", "2020-11-15", "2020-11-16", "2020-11-17", "2020-11-18", "2020-11-19", "2020-11-20", "2020-11-21", "2020-11-22", "2020-11-23", "2020-11-24", "2020-11-25", "2020-11-26", "2020-11-27", "2020-11-28", "2020-11-29", "2020-11-30", "2020-12-01", "2020-12-02", "2020-12-03", "2020-12-04", "2020-12-05", "2020-12-06", "2020-12-07", "2020-12-08", "2020-12-09", "2020-12-10", "2020-12-11", "2020-12-12", "2020-12-13", "2020-12-14", "2020-12-15", "2020-12-16", "2020-12-17", "2020-12-18", "2020-12-19",
                    "2020-12-20"
                 ]

    epi_dates = epi_dates[:config["num_train_dates"] + config["num_validation_dates"]]
    
    
    
    print("Loading data...")
    data_set = GraphRNN_dataset(epi_dates = epi_dates,
                                flow_dataset = flow_dataset,
                                epi_dataset = epi_dataset,
                                input_hor=config["input_hor"],
                                pred_hor=config["pred_hor"],
                                fake_data=False,
                                normalize_edge_weights=config["normalize_edge_weights"])

    train_data_set = torch.utils.data.Subset(data_set, range(config["num_train_dates"]+1-config["input_hor"]-config["pred_hor"]))
    val_data_set = torch.utils.data.Subset(data_set,
                                           range(config["num_train_dates"],
                                                 config["num_train_dates"] + config["num_validation_dates"] + 
                                                 1 - config["input_hor"] - config["pred_hor"]))
    
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=config["batch_size"], pin_memory=True, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=0)
    visualize = False
    if visualize:
        data_set.visualize(0, num_nodes=5, num_edges=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Data loaded.")
    input_edge_weights, input_node_data, target_edge_weights, target_node_data = next(iter(train_loader))
    fixed_edge_weights = input_edge_weights[0,0,:,:]
    
    from Neighbor_Agregation import *
    # neighbor_aggregator = Neighbor_Aggregation(n_nodes = data_set.n_nodes, h_size=config["h_size"],
    #                                            f_out_size=config["h_size"],fixed_edge_weights=fixed_edge_weights,
    #                                            device=device, dtype=torch.float32)
    # neighbor_aggregator = Neighbor_Aggregation_Simple(n_nodes = data_set.n_nodes, h_size=config["h_size"],
    #                                              f_out_size=config["h_size"],fixed_edge_weights=fixed_edge_weights,
    #                                              device=device, dtype=torch.float32)   
    neighbor_aggregator= Neighbor_Attention_multiHead(n_head=6,n_nodes=data_set.n_nodes,h_size=config["h_size"],
                                                    f_out_size = config["h_size"],edge_weights=fixed_edge_weights,device=device,dtype=torch.float32)
    #neighbor_aggregator=Neighbor_Attention_block(1,1,data_set.n_nodes,config["h_size"], config["h_size"],edge_weights=fixed_edge_weights, device=device)
    
    model = Graph_RNN_init(n_nodes = data_set.n_nodes,n_features=data_set.n_features,input_hor=config["input_hor"],num_params=25000,
                           fixed_edge_weights=fixed_edge_weights,neighbor_aggregator=neighbor_aggregator,device=device,dtype=torch.float32)
    
    # model  = Graph_RNN(n_nodes = data_set.n_nodes,
    #                    n_features = data_set.n_features,
    #                    h_size = config["h_size"],
    #                    f_out_size =config["h_size"],
    #                    input_hor = config["input_hor"],
    #                    fixed_edge_weights = fixed_edge_weights,
    #                    device=device,
    #                    dtype=torch.float32,
    #                    neighbor_aggregator=neighbor_aggregator
    #                    )
    
    


    
    def custom_loss(input_node_data, output, target_node_data):
        MSE = torch.nn.MSELoss().to(device)
        pred_hor = target_node_data.shape[1]
        input_hor = input_node_data.shape[1]
    
        loss = MSE(output[:,-pred_hor:,:,:], target_node_data[:,:pred_hor,:,:])
        loss += 0 * MSE(output[:,:input_hor,:,:], input_node_data[:,:,:])
        return loss
    criterion = custom_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["n_epochs"]/config["num_lr_steps"], gamma=config["lr_decay"])

    torch.autograd.set_detect_anomaly(False)

    model.to(device)
    profile = False
    if profile:
        log_dir = './log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Use torch.profiler to profile the model
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=False,
            with_stack=True
        ) as prof:
            print("Starting training with profiling...")
            train_losses, val_losses, parameter_mag, gradients, hidden_state_mag = train(model = model , train_loader = train_loader, val_loader = val_loader, criterion=
                        criterion, optimizer=optimizer,pred_hor=
                        config["pred_hor"], device=device, 
                        learning_rate_scheduler=learning_rate_scheduler,
                        n_epochs=config["n_epochs"],
                        max_grad_norm=config["max_grad_norm"],
                        save_name="test",
                        n_plots=5
                        )
            print("Finished training with profiling.")
            # Verify that the log directory is populated
        if os.listdir(log_dir):
            print(f"Log files generated in {log_dir}")
        else:
            print(f"No log files found in {log_dir}")
    else:
        train_losses, val_losses, parameter_mag, gradients, hidden_state_mag = train(model = model , train_loader = train_loader, val_loader = val_loader, criterion=
                        criterion, optimizer=optimizer,pred_hor=
                        config["pred_hor"], device=device, 
                        learning_rate_scheduler=learning_rate_scheduler,
                        n_epochs=config["n_epochs"],
                        max_grad_norm=config["max_grad_norm"],
                        save_name="test",
                        n_plots=5
                        )


    
    print("Plotting losses...")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig("Plots/losses_6heads.png")
    plt.show()
    
    print("Plotting parameter magnitudes...")
    plt.figure()
    for param_name, param_mag in parameter_mag.items():
       
       # if param_name.startswith("AG"):
            plt.plot(param_mag, label=param_name)
      
  #  plt.plot(hidden_state_mag, label="Hidden State")
    # for param_name, param_mag in parameter_mag.items():
    #     plt.plot(param_mag, label=param_name)
    # plt.plot(hidden_state_mag, label="Hidden State")
    
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Magnitude")
    # plt.yscale("log")
    plt.legend()
    plt.savefig("Plots/parameter_magnitude_6heads.png")
    plt.show()
    
    plt.figure()
    for param_name, grad in gradients["pre_limit"].items():
        #if param_name.startswith("AG"):
            plt.plot(grad, label=f"pre {param_name}" , linestyle="--")
    for param_name, grad in gradients["post_limit"].items():
       # if param_name.startswith("AG"):
            plt.plot(grad, label=f"post {param_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Magnitude")
    plt.legend()
    plt.yscale("log")
    plt.savefig("Plots/gradient_magnitude_6heads.png")
    plt.show()
    print("Finished training run.")
    
