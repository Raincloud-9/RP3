import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import pandas as pd
import networkx as nx
import numpy as np

# Load train and test CSV files into pandas DataFrame objects
train_df = pd.read_csv('VH.csv', header=None)
#test_df = pd.read_csv('80_KIAS_test_testing.csv', header=None)

# # Extract input and output columns for train and test datasets
# train_x_df = train_df.iloc[:, :9]
# train_y_df = train_df.iloc[:, 9:11]
# test_x_df = test_df.iloc[:, :9]
# test_y_df = test_df.iloc[:, 9:11]

# Extract input and output columns for train and test datasets
train_x_df = train_df.iloc[:75, :9]
train_y_df = train_df.iloc[:75, 9:11]
test_x_df = train_df.iloc[76:, :9]
test_y_df = train_df.iloc[76:, 9:11]

# Convert DataFrames to PyTorch tensors
train_x = (torch.tensor(train_x_df.values)).float()
train_y = (torch.tensor(train_y_df.values)).float()
test_x = (torch.tensor(test_x_df.values)).float()
test_y = (torch.tensor(test_y_df.values)).float()


# Normalize data
def normalize_tensor_max_min(tensor):
    max_values, _ = torch.max(torch.abs(tensor), dim=0)
    normalized_tensor = tensor / max_values
    return normalized_tensor

train_x = normalize_tensor_max_min(train_x)
train_y = normalize_tensor_max_min(train_y)
test_x = normalize_tensor_max_min(test_x)
test_y = normalize_tensor_max_min(test_y)


# def min_max_scaling(tensor):
#     min_vals = torch.min(tensor, dim=0).values
#     max_vals = torch.max(tensor, dim=0).values
#     return (tensor - min_vals) / (max_vals - min_vals)

# train_x = min_max_scaling(train_x)
# train_y = min_max_scaling(train_y)
# test_x = min_max_scaling(test_x)
# test_y = min_max_scaling(test_y)

# def normalize_z_score(tensor):
#     mean = torch.mean(tensor, dim=0)
#     std = torch.std(tensor, dim=0)
#     normalized_tensor = (tensor - mean) / std
#     return normalized_tensor

# #train_x = normalize_z_score(train_x)
# #train_y = normalize_z_score(train_y)
# #test_x = normalize_z_score(test_x)
# #test_y = normalize_z_score(test_y)

# # Different weight initialisations 

# import torch.nn.init as init

# def xavier_weight_init(m):
#     if isinstance(m, nn.Linear):
#         init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             init.zeros_(m.bias)

# def he_weight_init(m):
#     if isinstance(m, nn.Linear):
#         init.kaiming_uniform_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#             init.zeros_(m.bias)


def create_model():
    n_input, n_hidden, n_out = 9, 5, 2
    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          nn.Sigmoid(),
                          nn.Linear(n_hidden, n_out)
                          #nn.Sigmoid()
    )
    # Apply Xavier weight initialization
    #model.apply(xavier_weight_init)

    # Alternatively, apply He weight initialization
    # model.apply(he_weight_init)
    return model



def train_and_evaluate_model(model, train_x, train_y, test_x, test_y, epochs=1000, learning_rate=0.001):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # Train the model
        pred_y = model(train_x)
        train_loss = loss_function(pred_y, train_y)
        train_losses.append(train_loss.item())

        model.zero_grad()
        train_loss.backward()

        optimizer.step()

        # Test the model on the test dataset
        with torch.no_grad():
            test_pred_y = model(test_x)
            test_loss = loss_function(test_pred_y, test_y)
        test_losses.append(test_loss.item())

    return train_losses, test_losses



# same but regularized

def train_and_evaluate_model(model, train_x, train_y, test_x, test_y, epochs=3000, learning_rate=0.001, weight_decay=0.03):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # Train the model
        pred_y = model(train_x)
        train_loss = loss_function(pred_y, train_y)
        train_losses.append(train_loss.item())

        model.zero_grad()
        train_loss.backward()

        optimizer.step()

        # Test the model on the test dataset
        with torch.no_grad():
            test_pred_y = model(test_x)
            test_loss = loss_function(test_pred_y, test_y)
        test_losses.append(test_loss.item())


unpruned_model = create_model()
unpruned_train_losses, unpruned_test_losses = train_and_evaluate_model(unpruned_model, train_x, train_y, test_x, test_y)

    return train_losses, test_losses


# Create a copy of the trained model for pruning
import copy
pruned_model = copy.deepcopy(unpruned_model)

# Apply pruning
prune_amount = 0.90  # The fraction of connections to prune

layer1 = pruned_model[0]
prune.l1_unstructured(layer1, name="weight", amount=prune_amount)
layer2 = pruned_model[2]
prune.l1_unstructured(layer2, name="weight", amount=prune_amount)

# Train and evaluate the pruned model
pruned_train_losses, pruned_test_losses = train_and_evaluate_model(pruned_model, train_x, train_y, test_x, test_y)




# Plot the training and testing losses on the same graph
plt.plot(unpruned_train_losses, label='Training loss')
plt.plot(unpruned_test_losses, label='Testing loss')
plt.plot(pruned_train_losses, linestyle='--', color='C0', label='Pruned training loss')
plt.plot(pruned_test_losses, linestyle='--', color='C1', label='Pruned testing loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.title("Learning rate %f" % (learning_rate))
plt.legend()
plt.show()




# Access the pruning mask for the first layer's weights
layer1_weight_mask = layer1.weight_mask

# Access the pruning mask for the second layer's weights
layer2_weight_mask = layer2.weight_mask



total_layer1_weights = torch.zeros_like(layer1_weight_mask)
total_layer2_weights = torch.zeros_like(layer2_weight_mask)



total_layer1_weights += layer1_weight_mask
print("Layer 1 total weight mask:\n", total_layer1_weights)

total_layer2_weights += layer2_weight_mask
print("Layer 2 total weight mask:\n", total_layer2_weights)



def draw_pruned_nn(model, n_input, n_hidden, n_out):
    G = nx.DiGraph()

    # Add nodes for the input layer
    for i in range(n_input):
        G.add_node(f"Input_{i}", layer=0)

    # Add nodes for the hidden layer
    for i in range(n_hidden):
        G.add_node(f"Hidden_{i}", layer=1)

    # Add nodes for the output layer
    for i in range(n_out):
        G.add_node(f"Output_{i}", layer=2)

    # Add edges for the connections between input and hidden layers
    layer1_weight_mask = model[0].weight_mask.detach().numpy()
    for i in range(n_input):
        for j in range(n_hidden):
            if layer1_weight_mask[j, i] == 1:
                G.add_edge(f"Input_{i}", f"Hidden_{j}")

    # Add edges for the connections between hidden and output layers
    layer2_weight_mask = model[2].weight_mask.detach().numpy()
    for i in range(n_hidden):
        for j in range(n_out):
            if layer2_weight_mask[j, i] == 1:
                G.add_edge(f"Hidden_{i}", f"Output_{j}")

    # Draw the graph
    pos = nx.multipartite_layout(G, subset_key="layer", scale=3)
    for key, value in pos.items():
        pos[key] = (value[0], value[1] * 50)
    nx.draw(G, pos, with_labels=False, node_size=100)
    plt.show()

# Draw the pruned model's graph
draw_pruned_nn(pruned_model, 9, 5, 2)




# Trying to run 100 times in a row

num_iterations = 1000

for i in range(num_iterations):
    
    unpruned_model = create_model()
    unpruned_train_losses, unpruned_test_losses = train_and_evaluate_model(unpruned_model, train_x, train_y, test_x, test_y)

    # Create a new model and apply pruning
    pruned_model = copy.deepcopy(unpruned_model)

    # Apply pruning
    prune_amount = 0.90  # The fraction of connections to prune

    layer1 = pruned_model[0]
    prune.l1_unstructured(layer1, name="weight", amount=prune_amount)
    layer2 = pruned_model[2]
    prune.l1_unstructured(layer2, name="weight", amount=prune_amount)

    # Train and evaluate the pruned model
    #pruned_train_losses, pruned_test_losses = train_and_evaluate_model(pruned_model, train_x, train_y, test_x, test_y)

    # Access the pruning mask for the first layer's weights
    layer1_weight_mask = layer1.weight_mask

    # Access the pruning mask for the second layer's weights
    layer2_weight_mask = layer2.weight_mask

    total_layer1_weights += layer1_weight_mask
    total_layer2_weights += layer2_weight_mask

print("Layer 1 total weight mask:\n", total_layer1_weights)
print("Layer 2 total weight mask:\n", total_layer2_weights)


column_sums = torch.sum(total_layer1_weights, dim=0)
print(column_sums)

# Assuming column_sums is a PyTorch tensor
column_sums_np = column_sums.numpy()

# Create an array with indices starting from 1
indices = np.arange(1, len(column_sums_np) + 1)

# Plot a bar chart using the indices and the values in column_sums
plt.bar(indices, column_sums_np)
plt.xlabel('Input Neuron')
plt.ylabel('Number of connections maintained after pruning')
plt.title('Input Neuron Connections Retained After Pruning (1000 Iterations)')

# Set the x-axis labels to start from 1
plt.xticks(indices)

plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Assuming total_layer1_weights is your tensor
total_layer1_weights_np = total_layer1_weights.numpy()  # Convert tensor to NumPy array

plt.figure(figsize=(10, 8))  # Set the figure size
ax = sns.heatmap(total_layer1_weights_np, cmap='coolwarm', annot=True, fmt='.2f', vmin=0, vmax=127)

plt.title('Heatmap of retained connections between input and hidden layer')

# Set the tick labels to start from 1
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticks([x+0.5 for x in range(total_layer1_weights_np.shape[1])])
ax.set_yticks([y+0.5 for y in range(total_layer1_weights_np.shape[0])])
ax.set_xticklabels(range(1, total_layer1_weights_np.shape[1]+1))
ax.set_yticklabels(range(1, total_layer1_weights_np.shape[0]+1))

plt.xlabel('Input neurons')
plt.ylabel('Hidden layer neuron')
plt.show()
