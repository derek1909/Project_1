import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py
import ipdb
import yaml  # For logging training history to a YAML file

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

######################### Utility Classes #########################
# Loss function using Binary Cross Entropy with logits for classification
class Lossfunc(object):
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()

    def __call__(self, pred, target):
        return self.bce_loss(pred, target)

# This reads the matlab data from the .mat file provided
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_load(self):
        load = np.array(self.data['load_apply']).T
        return torch.tensor(load, dtype=torch.float32)

    def get_labels(self):
        labels = np.array(self.data['result']).T
        return torch.tensor(labels, dtype=torch.float32)

# Define data normalizer
class DataNormalizer(object):
    def __init__(self, data):
        data_flat = data.reshape(-1, data.shape[-1])
        self.mean = data_flat.mean(axis=0)
        self.std = data_flat.std(axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data_normalized):
        return data_normalized * self.std + self.mean

# Define the classifier neural network
class Classifier_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        Args:
            input_dim (int): Number of input features (in our case, 20 load points).
            hidden_dim (int): Number of hidden units.
        """
        super(Classifier_Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output a single logit for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, 1)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Flatten the 20 load points into a vector of size 20
        x_flat = x.view(x.size(0), -1)
        return self.model(x_flat)

######################### Experiment Loop #########################

# Hyperparameters
learning_rate = 3e-4
hidden_dim = 256
epochs = 500
batch_size = 100

# Create a folder for saving results
data_folder = 'Problem_2_student/Data'
results_folder = 'Problem_2_student/results/Classifier'
material_file = 'Eiffel_data.mat'
os.makedirs(results_folder, exist_ok=True)

# Generate synthetic data:
# 1000 samples, each with 20 load points (each point is 1-dimensional)
path = os.path.join(data_folder, material_file)
data_reader = MatRead(path)
load = data_reader.get_load()
labels = data_reader.get_labels()

[num_samples, num_load_points] = load.shape # 1000 x 20
print(load.shape)
print(labels.shape)

# Shuffle and split into training and test sets (80/20 split)
perm = torch.randperm(num_samples)
load = load[perm]
labels = labels[perm]
ntrain = int(0.8 * num_samples)
train_load = load[:ntrain]
train_labels = labels[:ntrain]
test_load = load[ntrain:]
test_labels = labels[ntrain:]

# Normalize load data using training data statistics
load_normalizer = DataNormalizer(train_load)
train_load_norm = load_normalizer.transform(train_load)
test_load_norm = load_normalizer.transform(test_load)

# Create DataLoader for training data
train_set = Data.TensorDataset(train_load_norm, train_labels)
train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Initialize the classifier network, loss function, optimizer, and (optional) scheduler
# The input dimension is 20 (load points flattened)
net = Classifier_Net(input_dim=num_load_points, hidden_dim=hidden_dim).to(device)
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Number of parameters in the classifier: {n_params}")

loss_func = Lossfunc()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.5)

loss_train_list = []
loss_test_list = []

# Training loop
for epoch in range(epochs):
    net.train()
    train_loss = 0.0
    for batch_load, batch_labels in train_loader:
        batch_load = batch_load.to(device)
        batch_labels = batch_labels.to(device)
        preds = net(batch_load)
        loss = loss_func(preds, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    loss_train_list.append(train_loss)

    # Compute test loss
    net.eval()
    with torch.no_grad():
        test_load_norm = test_load_norm.to(device)
        test_labels = test_labels.to(device)
        test_preds = net(test_load_norm)
        test_loss = loss_func(test_preds, test_labels).item()
    loss_test_list.append(test_loss)
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

print(f"Final Train Loss: {train_loss:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

# Log training history to a YAML file
history = {
    'epochs': epochs,
    'train_loss': loss_train_list,
    'test_loss': loss_test_list,
    'hyperparameters': {
        'learning_rate': learning_rate,
        'hidden_dim': hidden_dim,
        'batch_size': batch_size
    }
}
history_file = os.path.join(results_folder, 'training_history.yaml')
with open(history_file, 'w') as f:
    yaml.dump(history, f)
print(f"Training history logged to {history_file}")

# Plot and save train vs test loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_train_list, label='Train Loss')
plt.plot(range(epochs), loss_test_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
plt.legend()
plt.grid(True)
loss_plot_file = os.path.join(results_folder, 'train_vs_test_loss.png')
plt.savefig(loss_plot_file)
plt.close()
print(f"Loss plot saved to {loss_plot_file}")

# Evaluate test accuracy
net.eval()
with torch.no_grad():
    test_preds = net(test_load_norm.to(device))
    # Apply sigmoid to convert logits to probabilities, then threshold at 0.5
    predicted_labels = (torch.sigmoid(test_preds) > 0.5).float()
    accuracy = (predicted_labels.cpu() == test_labels.cpu()).float().mean().item()

# Evaluate test accuracy and return wrong trial indices
net.eval()
with torch.no_grad():
    test_preds = net(test_load_norm.to(device))
    # Apply sigmoid and threshold to get predicted labels
    predicted_labels = (torch.sigmoid(test_preds) > 0.5).float()
    accuracy = (predicted_labels.cpu() == test_labels.cpu()).float().mean().item()
    # Find indices where predicted labels do not match true labels
    wrong_indices = (predicted_labels.cpu() != test_labels.cpu()).nonzero(as_tuple=True)[0]
print("Wrong trial indices:", wrong_indices.tolist())
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Optionally, plot a scatter of predicted probabilities vs true labels
with torch.no_grad():
    test_probs = torch.sigmoid(test_preds).cpu().numpy().flatten()
    true_labels = test_labels.cpu().numpy().flatten()
plt.figure(figsize=(8, 5))
plt.scatter(range(len(test_probs)), test_probs, label='Predicted Probability', color='b', alpha=0.6)
plt.scatter(range(len(true_labels)), true_labels, label='True Label', color='r', marker='x')
plt.xlabel('Test Sample Index')
plt.ylabel('Probability / Label')
plt.title('Predicted Probabilities vs True Labels')
plt.legend()
plt.grid(True)
prob_plot_file = os.path.join(results_folder, 'predicted_vs_true.png')
plt.savefig(prob_plot_file)
plt.close()
print(f"Predicted vs True plot saved to {prob_plot_file}")

print("All experiments completed.")