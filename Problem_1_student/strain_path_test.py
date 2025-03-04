import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#############################################
# 1. Define the network architectures #
#############################################
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


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        """
        Args:
            input_dim: Number of input features (e.g., strain components).
            hidden_dim: Hidden state dimension.
            output_dim: Number of output features (if None, equals input_dim).
        """
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        # f: update hidden state
        self.f = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # g: compute output (stress) from current input and hidden state
        self.g = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size, T, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            h = h + self.f(torch.cat([x_t, h], dim=1))
            y_t = self.g(torch.cat([x_t, h], dim=1))
            outputs.append(y_t.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs

##########################################
# 2. Load normalization parameters       #
##########################################
def load_normalization_params(file_path):
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

# Define result folder (where your trained model and normalization parameters are stored)
result_folder = 'Problem_1_student/results/abc/Material_B'
norm_file = os.path.join(result_folder, 'normalization_params.yaml')
model_file = os.path.join(result_folder, 'trained_model.pth')

norm_params = load_normalization_params(norm_file)
# Extract strain and stress normalization parameters (as numpy arrays)
strain_mean = np.array(norm_params['strain']['mean'])
strain_std  = np.array(norm_params['strain']['std'])
stress_mean = np.array(norm_params['stress']['mean'])
stress_std  = np.array(norm_params['stress']['std'])

ndim = len(strain_mean)

# Infer number of input dimensions from strain parameters

config={
    "learning_rate": 3e-3,
    "hidden_dim": 8,
    "epochs": 420,
    "batch_size": 50,
    "sche_step_size": 65,
    "hidden_layer": 1,
    "model": "FCNN"
}


# config={
#     "learning_rate": 3e-3,
#     "hidden_dim": 64,
#     "epochs": 500,
#     "batch_size": 50,
#     "sche_step_size": 100,
#     "hidden_layer": 4,
#     "model": "FCNN"
# }

# config={
#     "learning_rate": 3e-3,
#     "hidden_dim": 8,
#     "epochs": 500,
#     "batch_size": 50,
#     "sche_step_size": 100,
#     "hidden_layer": 1,
#     "model": "RNN"
# }

learning_rate = config['learning_rate']
hidden_dim = config['hidden_dim']
epochs = config['epochs']
batch_size = config['batch_size']
sche_step_size = config['sche_step_size']
hidden_layer = config['hidden_layer']
data_folder = 'Problem_1_student/Data'
results_base = 'Problem_1_student/results/abc'


# Define units (adjust as needed)
strain_unit = "mm/mm"  # e.g., unitless strain or mm/mm
stress_unit = "MPa"      # e.g., stress in megapascals

##############################################
# 3. Generate a test strain signal manually  #
##############################################
# Let's assume a time-series with nstep steps.
nstep = 50
# Create a time axis (0 to 1)
time_steps = np.linspace(0, 1, nstep)
# Generate a signal: first linearly increasing then decreasing.
ramp_up   = np.linspace(0, 1, nstep//2)
ramp_down = np.linspace(1, 0, nstep - nstep//2)
ramp = np.concatenate((ramp_up, ramp_down))
# For simplicity, use the same ramp for all strain components.
# Shape: (1, nstep, ndim)
test_strain = np.tile(ramp.reshape(1, nstep, 1), (1, 1, ndim)).astype(np.float32) * 0.04

##########################################
# 4. Normalize the test strain           #
##########################################
# Normalization: (data - mean) / std. Use broadcasting across the last dimension.
test_strain_norm = (test_strain - strain_mean.reshape(1,1,ndim)) / strain_std.reshape(1,1,ndim)
# Convert to torch tensor on the proper device.
test_strain_norm = torch.tensor(test_strain_norm, dtype=torch.float32, device=device)

##########################################
# 5. Load the trained network and get predicted stress
##########################################
# For an RNN, output_dim is usually the same as input_dim (stress components match strain components)
# net = RNN(input_dim=ndim, hidden_dim=hidden_dim, output_dim=ndim).to(device)
net = FCNN(input_dim=ndim, hidden_dim=hidden_dim, output_dim=ndim, num_layers=hidden_layer).to(device)
net.load_state_dict(torch.load(model_file, map_location=device))
net.eval()

with torch.no_grad():
    pred_stress_norm = net(test_strain_norm)  # still normalized
# Convert prediction to numpy array (shape: (1, nstep, ndim))
pred_stress_norm = pred_stress_norm.cpu().numpy()

##########################################
# 6. Denormalize the predicted stress     #
##########################################
# Denormalization: data * std + mean
pred_stress = pred_stress_norm * stress_std.reshape(1,1,ndim) + stress_mean.reshape(1,1,ndim)

##########################################
# 7. Plot the results                     #
##########################################
# Plot the test strain (original scale)
plt.figure(figsize=(6,6))
for comp in range(ndim):
    plt.plot(time_steps, test_strain[0,:,comp], label=f'Strain comp {comp}')
plt.xlabel('Time')
plt.ylabel('Strain (mm/mm)')  # Adjust unit if needed
# plt.title('Test Strain (Increasing then Decreasing)')
plt.legend()
plt.grid(True)
plt.tight_layout()
strain_plot_file = os.path.join(result_folder, 'test_strain.png')
plt.savefig(strain_plot_file)
plt.close()
print("Test strain plot saved to:", strain_plot_file)

# Plot the predicted stress vs time
plt.figure(figsize=(6,6))
for comp in range(ndim):
    plt.plot(time_steps, pred_stress[0,:,comp], label=f'Predicted Stress comp {comp}')
plt.xlabel('Time')
plt.ylabel(f'Stress ({stress_unit})')
# plt.title('Predicted Stress')
plt.legend()
plt.grid(True)
plt.tight_layout()
stress_plot_file = os.path.join(result_folder, 'predicted_test_stress.png')
plt.savefig(stress_plot_file)
plt.close()
print("Predicted test stress plot saved to:", stress_plot_file)

# Optionally, plot Stress vs Strain for each component (denormalized values)
plt.figure(figsize=(6,6))
for comp in range(ndim):
    plt.plot(test_strain[0,:,comp], pred_stress[0,:,comp], label=f'Component {comp}')
plt.xlabel('Strain (mm/mm)')
plt.ylabel(f'Stress ({stress_unit})')
plt.title('Predicted Stress vs Test Strain')
plt.legend()
plt.grid(True)
plt.tight_layout()
ss_plot_file = os.path.join(result_folder, 'stress_vs_strain.png')
plt.savefig(ss_plot_file)
plt.close()
print("Stress vs Strain plot saved to:", ss_plot_file)