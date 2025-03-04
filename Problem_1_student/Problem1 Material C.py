import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py
import yaml
# import wandb  # Import wandb
# import shutil


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize wandb run
# wandb.init(
config={
    "learning_rate": 3e-3,
    "hidden_dim": 64,
    "epochs": 500,
    "batch_size": 50,
    "sche_step_size": 100,
    "hidden_layer": 4,
    "model": "FCNN"
}
# )
# config = wandb.config

######################### Utility Classes #########################
# Loss function using MSE of stress
class Lossfunc(object):
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred_stress, target_stress):
        return self.mse_loss(pred_stress, target_stress)

# This reads the matlab data from the .mat file provided
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_strain(self):
        strain = np.array(self.data['strain']).transpose(2,0,1)
        return torch.tensor(strain, dtype=torch.float32)

    def get_stress(self):
        stress = np.array(self.data['stress']).transpose(2,0,1)
        return torch.tensor(stress, dtype=torch.float32)

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

# Define the neural network for the constitutive model
class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None, num_layers=4):
        """
        Args:
            input_dim (int): Number of input features (ndim).
            hidden_dim (int): Size of the hidden layers.
            output_dim (int, optional): Number of output features.
                                        If None, set equal to input_dim.
            num_layers (int): Number of linear layers in the network.
        """
        super(FCNN, self).__init__()
        if output_dim is None:
            output_dim = input_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, nstep, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, nstep, output_dim)
        """
        batch_size, nstep, input_dim = x.shape
        # Flatten the time dimension with the batch dimension so that each time step is processed individually.
        x_flat = x.view(batch_size * nstep, input_dim)
        output = self.model(x_flat)
        # Reshape back to (batch_size, nstep, output_dim)
        out = output.view(batch_size, nstep, -1)
        return out

# RNN Architecture for Constitutive Modeling
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        """
        Args:
            input_dim: Dimension of the input (e.g., number of strain components).
            hidden_dim: Dimension of the hidden state (represents the memory capacity).
            output_dim: Dimension of the output (e.g., number of stress components).
        """
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # f: Function to update the hidden state.
        # It takes the concatenation of the current input and previous hidden state.
        self.f = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # g: Function to compute the output (predicted stress) from the current input and hidden state.
        self.g = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass for a sequence of inputs.
        Args:
            x: Input tensor of shape [batch, T, input_dim] where T is the number of time steps.
        Returns:
            outputs: Output tensor of shape [batch, T, output_dim] (predicted stress over time).
        """
        batch_size, T, _ = x.shape
        # Initialize hidden state as zeros
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs = []
        
        # Process each time step sequentially
        for t in range(T):
            x_t = x[:, t, :]  # current input at time step t, shape [batch, input_dim]
            # Update hidden state with a residual update (similar to forward Euler discretization)
            h = h + self.f(torch.cat([x_t, h], dim=1))
            # Compute the output at current time step
            y_t = self.g(torch.cat([x_t, h], dim=1))
            outputs.append(y_t.unsqueeze(1))  # Add time dimension
            
        # Concatenate the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
######################### Experiment Loop #########################
material_file = 'Material_C.mat'
learning_rate = config['learning_rate']
hidden_dim = config['hidden_dim']
epochs = config['epochs']
batch_size = config['batch_size']
sche_step_size = config['sche_step_size']
hidden_layer = config['hidden_layer']
data_folder = 'Problem_1_student/Data'
results_base = 'Problem_1_student/results/abc'
os.makedirs(results_base, exist_ok=True)

material_name = os.path.splitext(material_file)[0]
print(f"Processing {material_name}...")

result_folder = os.path.join(results_base, material_name)
os.makedirs(result_folder, exist_ok=True)

# Read data
path = os.path.join(data_folder, material_file)
data_reader = MatRead(path)
strain = data_reader.get_strain()
stress = data_reader.get_stress()

# Shuffle and split into training and test sets
torch.manual_seed(43)

n_samples = strain.shape[0]
perm = torch.randperm(n_samples)
strain = strain[perm]
stress = stress[perm]
ntrain = int(0.8 * n_samples)
train_strain = strain[:ntrain]
train_stress = stress[:ntrain]
test_strain = strain[ntrain:]
test_stress = stress[ntrain:]

# Normalize using training data
strain_normalizer = DataNormalizer(train_strain)
train_strain_encode = strain_normalizer.transform(train_strain)
test_strain_encode = strain_normalizer.transform(test_strain)

stress_normalizer = DataNormalizer(train_stress)
train_stress_encode = stress_normalizer.transform(train_stress)
test_stress_encode = stress_normalizer.transform(test_stress)

# Define units (adjust as needed)
strain_unit = "mm/mm"  # e.g., unitless strain or mm/mm
stress_unit = "MPa"      # e.g., stress in megapascals

# Save normalization parameters for strain and stress
normalization_params = {
    'strain': {
        'mean': strain_normalizer.mean.tolist(),
        'std':  strain_normalizer.std.tolist()
    },
    'stress': {
        'mean': stress_normalizer.mean.tolist(),
        'std':  stress_normalizer.std.tolist()
    }
}
normalization_file = os.path.join(result_folder, 'normalization_params.yaml')
with open(normalization_file, 'w') as f:
    yaml.dump(normalization_params, f)
print(f"Normalization parameters saved to {normalization_file}")

ndim = strain.shape[2]
nstep = strain.shape[1]
dt = 1 / (nstep - 1)

# Create DataLoader for training data
train_set = Data.TensorDataset(train_strain_encode, train_stress_encode)
train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Initialize the network based on the wandb config parameter "model"
if config['model'] == "RNN":
    net = RNN(input_dim=ndim, hidden_dim=hidden_dim, output_dim=ndim).to(device)
elif config['model'] == "FCNN":
    net = FCNN(input_dim=ndim, hidden_dim=hidden_dim, output_dim=ndim, num_layers=hidden_layer).to(device)
else:
    raise ValueError(f"Unknown model type specified in config: {config['model']}")

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Number of parameters for {material_name}, {config['model']}: {n_params}")

loss_func = Lossfunc()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sche_step_size, gamma=0.5)

# Watch the model with wandb to log gradients and parameters
# wandb.watch(net, log="all", log_freq=100)

loss_train_list = []
loss_test_list = []

# Training loop
for epoch in range(epochs):
    net.train()
    trainloss = 0.0
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = net(inputs)
        loss = loss_func(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
    trainloss /= len(train_loader)
    loss_train_list.append(trainloss)

    # Compute test loss
    net.eval()
    with torch.no_grad():
        test_strain_encode = test_strain_encode.to(device)
        test_stress_encode = test_stress_encode.to(device)
        output_test = net(test_strain_encode)
        testloss = loss_func(output_test, test_stress_encode).item()
    loss_test_list.append(testloss)
    scheduler.step()

    # Log metrics to wandb
    # wandb.log({
    #     "epoch": epoch,
    #     "train_loss": trainloss,
    #     "test_loss": testloss
    # })

    if epoch % 10 == 0:
        print(f"{material_name} - epoch: {epoch}, train loss: {trainloss:.6f}, test loss: {testloss:.6f}")

print(f"{material_name} - Final Train loss: {trainloss:.6f}")
print(f"{material_name} - Final Test loss: {testloss:.6f}")

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
history_file = os.path.join(result_folder, 'training_history.yaml')
with open(history_file, 'w') as f:
    yaml.dump(history, f)
print(f"Training history logged to {history_file}")

# Plot and save loss curve in log scale
plt.figure(figsize=(6, 6))
plt.plot(range(epochs), loss_train_list, label='Train Loss')
plt.plot(range(epochs), loss_test_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # Set y-axis to log scale
plt.title(f'{material_name}: Train vs Test Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_file = os.path.join(result_folder, 'train_vs_test_loss.png')
plt.savefig(loss_plot_file)
plt.close()
print(f"Loss plot saved to {loss_plot_file}")

# Log loss plot image to wandb
# wandb.log({"loss_plot": wandb.Image(loss_plot_file)})

# Plot one sample of truth stress vs predicted stress
net.eval()
# Choose a fixed sample (here, sample index 0)
with torch.no_grad():
    sample_input_norm = test_strain_encode[0:1].to(device)
    sample_target_norm = test_stress_encode[0:1].to(device)
    sample_output_norm = net(sample_input_norm) # Predictions (normalized)

# Denormalize (convert normalized data back to original scale)
sample_input_unnorm = strain_normalizer.inverse_transform(sample_input_norm.cpu())
sample_target_unnorm = stress_normalizer.inverse_transform(sample_target_norm.cpu())
sample_output_unnorm = stress_normalizer.inverse_transform(sample_output_norm.cpu())

# Create a time axis for plotting
time_steps = np.linspace(0, dt*(nstep-1), nstep)

# Plot "Stress Comparison (Sample 0)" with units in labels
plt.figure(figsize=(6, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for comp in range(ndim):
    plt.plot(time_steps, sample_target_unnorm[0, :, comp], 
             label=f'Truth Stress comp {comp}', color=colors[comp % len(colors)])
    plt.plot(time_steps, sample_output_unnorm[0, :, comp], '--', 
             label=f'Predicted Stress', color=colors[comp % len(colors)])
plt.xlabel(f'Time')
plt.ylabel(f'Stress ({stress_unit})')
plt.title(f'{material_name}: Stress Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
sample_plot_file = os.path.join(result_folder, 'stress_comparison_sample.png')
plt.savefig(sample_plot_file)
plt.close()
print(f"Stress comparison plot saved to {sample_plot_file}")

# Plot "True Stress vs Strain (Sample 0)" with units in labels
plt.figure(figsize=(6, 6))
for comp in range(ndim):
    plt.plot(sample_input_unnorm[0, :, comp], sample_target_unnorm[0, :, comp],
             label=f'True Stress vs Strain comp {comp}')
    plt.plot(sample_input_unnorm[0, :, comp], sample_output_unnorm[0, :, comp],
             label=f'Predicted Stress vs Strain')
plt.xlabel(f'Strain ({strain_unit})')
plt.ylabel(f'Stress ({stress_unit})')
plt.title(f'{material_name}: Stress vs Strain')
plt.legend()
plt.grid(True)
plt.tight_layout()
fig_file = os.path.join(result_folder, 'stress_vs_strain_sample.png')
plt.savefig(fig_file)
plt.close()
print(f"Stress vs strain plot saved to {fig_file}")

model_path = os.path.join(result_folder, 'trained_model.pth')
torch.save(net.state_dict(), model_path)
print("Saved model to:", model_path)

# Finish the wandb run
# wandb.finish()