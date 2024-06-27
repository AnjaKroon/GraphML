import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import pandas as pd
from tqdm import tqdm
from src.utils.dataloaders.Split_graph_data import GraphRNN_dataset
from GraphRNN import Graph_RNN, Neighbor_Aggregation
import matplotlib.pyplot as plt
import json
import torch.profiler
import numpy as np


def train(model, train_loader, val_loader, criterion, optimizer, pred_hor, device,
          learning_rate_scheduler, n_epochs =10, save_string=None, max_grad_norm=1, n_plots=None):
    train_losses = []
    val_losses = []
    parameter_mag = {param_name: [] for param_name, param in model.named_parameters()}
    gradients = {}
    gradients["pre_limit"] = {param_name: [] for param_name, param in model.named_parameters()}
    gradients["post_limit"] = {param_name: [] for param_name, param in model.named_parameters()}
    hidden_states = []
    input_hor = model.input_hor

    for epoch in tqdm(range(n_epochs), desc="Epoch"):
        save_string_epoch= f"{save_string}_epoch_{epoch}"
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
        
        if epoch % (n_epochs/n_epochs) == 0:
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
                plt.savefig(f"GraphRNN\plots\covid_predictions\plot_{save_string}_{epoch}.png")
                plt.close()
            
        learning_rate_scheduler.step()
            
    if save_string_epoch is not None:
        torch.save(model.state_dict(), f"GraphRNN\models\model_state_dict_{save_string}.pth")
        with open(f"GraphRNN\losses\losses_{save_string}.json", "w") as f:
            json.dump(train_losses, f)
        with open(f"GraphRNN/losses/val_losses_{save_string}.json", "w") as f:
            json.dump(val_losses, f)
    return train_losses, val_losses, parameter_mag, gradients, hidden_states