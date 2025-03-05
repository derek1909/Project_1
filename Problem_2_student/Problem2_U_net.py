import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py
import ipdb
import yaml  # For logging training history to a YAML file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        super(MatRead, self).__init__()
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

######################### U-Net for 1D Data #########################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128]):
        super(UNet1D, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # Encoder path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # Decoder path
        self.ups = nn.ModuleList()
        rev_features = features[::-1]
        for feature in rev_features:
            self.ups.append(nn.ConvTranspose1d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape[-1] != skip_connection.shape[-1]:
                x = nn.functional.pad(x, (0, skip_connection.shape[-1] - x.shape[-1]))
            x = torch.cat([skip_connection, x], dim=1)
            x = self.ups[idx+1](x)
        return x

class UNetClassifier(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128], num_classes=1):
        super(UNetClassifier, self).__init__()
        self.unet = UNet1D(in_channels, features)
        # 使用全局平均池化将特征聚合，再接一个全连接层输出单个logit
        self.classifier = nn.Linear(features[0], num_classes)
    def forward(self, x):
        # x: (batch, 1, length)
        features = self.unet(x)   # 输出 shape: (batch, features[0], length)
        gap = features.mean(dim=2)  # 全局平均池化, shape: (batch, features[0])
        out = self.classifier(gap)  # shape: (batch, num_classes)
        return out

######################### Experiment Loop #########################

# Hyperparameters
learning_rate = 1e-4
feature_size = [64, 128]
epochs = 500
batch_size = 1000

# Create a folder for saving results
data_folder = 'Problem_2_student/Data'
results_folder = 'Problem_2_student/results/U-Net'
material_file = 'Eiffel_data2.mat'
os.makedirs(results_folder, exist_ok=True)

# Generate synthetic data:
# 1000 samples, each with 20 load points (each point is 1-dimensional)
path = os.path.join(data_folder, material_file)
data_reader = MatRead(path)
load = data_reader.get_load()    # shape: (1000, 20)
labels = data_reader.get_labels()  # shape: (1000, 1)

[num_samples, num_load_points] = load.shape

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

# Add a channel dimension for UNetClassifier (batch, channel, length)
train_load_norm = train_load_norm.unsqueeze(1)  # (ntrain, 1, 20)
test_load_norm = test_load_norm.unsqueeze(1)    # (num_test, 1, 20)

# Create DataLoader for training data
train_set = Data.TensorDataset(train_load_norm, train_labels)
train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Initialize the U-Net based classifier network, loss function, optimizer, and scheduler
net = UNetClassifier(in_channels=1, features=feature_size, num_classes=1).to(device)
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Number of parameters in the U-Net: {n_params}")

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
        'features': feature_size,
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

# Evaluate test accuracy and return wrong trial indices
net.eval()
with torch.no_grad():
    test_preds = net(test_load_norm.to(device))
    # Apply sigmoid to convert logits to probabilities, then threshold at 0.5
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
plt.figure(figsize=(5, 5))
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