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
# 1. Define the network architectures      #
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
        batch_size, nstep, input_dim = x.shape
        x_flat = x.view(batch_size * nstep, input_dim)
        output = self.model(x_flat)
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
result_folder = 'Problem_1_student/results/abc/Material_C_RNN'
norm_file = os.path.join(result_folder, 'normalization_params.yaml')
model_file = os.path.join(result_folder, 'trained_model.pth')

norm_params = load_normalization_params(norm_file)
# Extract strain and stress normalization parameters (as numpy arrays)
strain_mean = np.array(norm_params['strain']['mean'])
strain_std  = np.array(norm_params['strain']['std'])
stress_mean = np.array(norm_params['stress']['mean'])
stress_std  = np.array(norm_params['stress']['std'])

ndim = len(strain_mean)
print("Detected input dimension (ndim):", ndim)

##########################################
# 3. Set configuration and units         #
##########################################
config = {
    "learning_rate": 3e-3,
    "hidden_dim": 8,
    "epochs": 500,
    "batch_size": 50,
    "sche_step_size": 100,
    "hidden_layer": 1,
    "model": "RNN"
}

learning_rate = config['learning_rate']
hidden_dim = config['hidden_dim']
epochs = config['epochs']
batch_size = config['batch_size']
sche_step_size = config['sche_step_size']
hidden_layer = config['hidden_layer']
data_folder = 'Problem_1_student/Data'

# Define units (adjust as needed)
strain_unit = "mm/mm"
stress_unit = "MPa"

##########################################
# 4. Generate a test strain signal manually  #
##########################################
nstep = 50
time_steps = np.linspace(0, 1, nstep)
# Create a ramp-up then ramp-down waveform
ramp_up   = np.linspace(0, 1, nstep//2)
ramp_down = np.linspace(1, 0, nstep - nstep//2)
ramp = np.concatenate((ramp_up, ramp_down))
# Base test strain: shape (1, nstep, ndim)
test_strain = np.tile(ramp.reshape(1, nstep, 1), (1, 1, ndim)).astype(np.float32)

##########################################
# 5. Normalize the test strain           #
##########################################
test_strain_norm = (test_strain - strain_mean.reshape(1,1,ndim)) / strain_std.reshape(1,1,ndim)
test_strain_norm = torch.tensor(test_strain_norm, dtype=torch.float32, device=device)

##########################################
# 6. Load the trained network and predict stress
##########################################
net = RNN(input_dim=ndim, hidden_dim=hidden_dim, output_dim=ndim).to(device)
net.load_state_dict(torch.load(model_file, map_location=device))
net.eval()

with torch.no_grad():
    pred_stress_norm = net(test_strain_norm)
pred_stress_norm = pred_stress_norm.cpu().numpy()
pred_stress = pred_stress_norm * stress_std.reshape(1,1,ndim) + stress_mean.reshape(1,1,ndim)

##########################################
# 7. Plot the original results           #
##########################################
# Plot the test strain (denormalized, already in original scale)
plt.figure(figsize=(6,6))
for comp in range(ndim):
    plt.plot(time_steps, test_strain[0,:,comp], label=f'Strain comp {comp}')
plt.xlabel('Time')
plt.ylabel(f'Strain ({strain_unit})')
plt.legend()
plt.grid(True)
plt.tight_layout()
strain_plot_file = os.path.join(result_folder, 'test_strain.png')
plt.savefig(strain_plot_file)
plt.close()
print("Test strain plot saved to:", strain_plot_file)

# Plot the predicted stress vs time
# plt.figure(figsize=(6,6))
# for comp in range(ndim):
#     plt.plot(time_steps, pred_stress[0,:,comp], label=f'Predicted Stress comp {comp}')
# plt.xlabel('Time')
# plt.ylabel(f'Stress ({stress_unit})')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# stress_plot_file = os.path.join(result_folder, 'predicted_test_stress.png')
# plt.savefig(stress_plot_file)
# plt.close()
# print("Predicted test stress plot saved to:", stress_plot_file)

# # Plot Stress vs Strain for each component
# plt.figure(figsize=(6,6))
# for comp in range(ndim):
#     plt.plot(test_strain[0,:,comp], pred_stress[0,:,comp], label=f'Component {comp}')
# plt.xlabel(f'Strain ({strain_unit})')
# plt.ylabel(f'Stress ({stress_unit})')
# plt.title('Predicted Stress vs Test Strain')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# ss_plot_file = os.path.join(result_folder, 'stress_vs_strain.png')
# plt.savefig(ss_plot_file)
# plt.close()
# print("Stress vs Strain plot saved to:", ss_plot_file)

##########################################
# 8. Multiple Amplitude Test             #
##########################################
# Define a list of amplitude factors to test
amplitudes = [0.01, 0.05, 0.1, 0.2, 0.8]

# Plot predicted stress for different amplitudes (first stress component)
plt.figure(figsize=(6,6))
for amp in amplitudes:
    # Multiply base test strain by amplitude
    test_strain_amp = test_strain * amp
    # Normalize the amplified test strain
    test_strain_amp_norm = (test_strain_amp - strain_mean.reshape(1,1,ndim)) / strain_std.reshape(1,1,ndim)
    test_strain_amp_norm = torch.tensor(test_strain_amp_norm, dtype=torch.float32, device=device)
    # Get prediction from the network
    with torch.no_grad():
        pred_stress_amp_norm = net(test_strain_amp_norm)
    pred_stress_amp_norm = pred_stress_amp_norm.cpu().numpy()
    pred_stress_amp = pred_stress_amp_norm * stress_std.reshape(1,1,ndim) + stress_mean.reshape(1,1,ndim)
    # Plot predicted stress for the first component
    plt.plot(time_steps, pred_stress_amp[0,:,0], label=f'Max strain {amp}')
plt.xlabel('Time')
plt.ylabel(f'Stress ({stress_unit})')
# plt.title('Predicted Stress vs Time for Different Amplitudes (Component 0)')
plt.legend()
plt.grid(True)
plt.tight_layout()
multi_amp_plot_file = os.path.join(result_folder, 'predicted_stress_multi_amplitude.png')
plt.savefig(multi_amp_plot_file)
plt.close()
print("Multi-amplitude predicted stress plot saved to:", multi_amp_plot_file)

# NEW: Plot test strain signals for different amplitudes (first strain component)
plt.figure(figsize=(6,6))
for amp in amplitudes:
    # Multiply base test strain by amplitude
    test_strain_amp = test_strain * amp
    # Plot the test strain for the first component
    plt.plot(time_steps, test_strain_amp[0,:,0], label=f'Max strain {amp} MPa')
plt.xlabel('Time')
plt.ylabel(f'Strain ({strain_unit})')
# plt.title('Test Strain vs Time for Different Amplitudes (Component 0)')
plt.legend()
plt.grid(True)
plt.tight_layout()
multi_amp_strain_plot_file = os.path.join(result_folder, 'test_strain_multi_amplitude.png')
plt.savefig(multi_amp_strain_plot_file)
plt.close()
print("Multi-amplitude test strain plot saved to:", multi_amp_strain_plot_file)


##########################################
# 9. Step Strain Test (Single Step)
##########################################
# Create a step strain waveform: first half zero, second half constant
step_value = 1.0
step_strain = np.zeros((1, nstep, ndim), dtype=np.float32)
step_strain[0, 15:] = step_value

# Normalize the step strain
step_strain_norm = (step_strain - strain_mean.reshape(1,1,ndim)) / strain_std.reshape(1,1,ndim)
step_strain_norm = torch.tensor(step_strain_norm, dtype=torch.float32, device=device)

# Predict stress for the step strain
with torch.no_grad():
    pred_step_stress_norm = net(step_strain_norm)
pred_step_stress_norm = pred_step_stress_norm.cpu().numpy()
pred_step_stress = pred_step_stress_norm * stress_std.reshape(1,1,ndim) + stress_mean.reshape(1,1,ndim)

# Plot the step strain signal (first strain component)
plt.figure(figsize=(6,6))
for comp in range(ndim):
    plt.plot(time_steps, step_strain[0,:,comp], label=f'Step Strain comp {comp}')
plt.xlabel('Time')
plt.ylabel(f'Strain ({strain_unit})')
plt.title('Step Strain vs Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
step_strain_plot_file = os.path.join(result_folder, 'step_strain.png')
plt.savefig(step_strain_plot_file)
plt.close()
print("Step strain plot saved to:", step_strain_plot_file)

# Plot predicted stress for step strain (first stress component)
plt.figure(figsize=(6,6))
plt.plot(time_steps, pred_step_stress[0,:,0], label='Predicted Stress (Step)')
plt.xlabel('Time')
plt.ylabel(f'Stress ({stress_unit})')
plt.title('Predicted Stress vs Time for Step Strain (Component 0)')
plt.legend()
plt.grid(True)
plt.tight_layout()
step_stress_plot_file = os.path.join(result_folder, 'predicted_step_stress.png')
plt.savefig(step_stress_plot_file)
plt.close()
print("Predicted step stress plot saved to:", step_stress_plot_file)

##########################################
# 10. Multiple Amplitude Test (Step Strain)
##########################################
# Define a list of step amplitude factors to test
step_amplitudes = [0.01, 0.05, 0.1, 0.2, 0.8]

# Plot predicted stress (first stress component) for different step amplitudes
plt.figure(figsize=(6,6))
for amp in step_amplitudes:
    step_strain_amp = np.zeros((1, nstep, ndim), dtype=np.float32)
    step_strain_amp[0, 15:] = amp  # step amplitude is amp
    step_strain_amp_norm = (step_strain_amp - strain_mean.reshape(1,1,ndim)) / strain_std.reshape(1,1,ndim)
    step_strain_amp_norm = torch.tensor(step_strain_amp_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_step_stress_amp_norm = net(step_strain_amp_norm)
    pred_step_stress_amp_norm = pred_step_stress_amp_norm.cpu().numpy()
    pred_step_stress_amp = pred_step_stress_amp_norm * stress_std.reshape(1,1,ndim) + stress_mean.reshape(1,1,ndim)
    plt.plot(time_steps, pred_step_stress_amp[0,:,0], label=f'Step Amplitude {amp}')
plt.xlabel('Time')
plt.ylabel(f'Stress ({stress_unit})')
plt.title('Predicted Stress vs Time for Different Step Amplitudes (Component 0)')
plt.legend()
plt.grid(True)
plt.tight_layout()
step_amp_stress_plot_file = os.path.join(result_folder, 'predicted_step_stress_multi_amplitude.png')
plt.savefig(step_amp_stress_plot_file)
plt.close()
print("Multi-amplitude predicted step stress plot saved to:", step_amp_stress_plot_file)

# Plot test step strain signals for different amplitudes (first strain component)
plt.figure(figsize=(6,6))
for amp in step_amplitudes:
    step_strain_amp = np.zeros((1, nstep, ndim), dtype=np.float32)
    step_strain_amp[0, nstep//2:] = amp
    plt.plot(time_steps, step_strain_amp[0,:,0], label=f'Step Amplitude {amp}')
plt.xlabel('Time')
plt.ylabel(f'Strain ({strain_unit})')
plt.title('Step Strain vs Time for Different Amplitudes (Component 0)')
plt.legend()
plt.grid(True)
plt.tight_layout()
step_amp_strain_plot_file = os.path.join(result_folder, 'step_strain_multi_amplitude.png')
plt.savefig(step_amp_strain_plot_file)
plt.close()
print("Multi-amplitude step strain plot saved to:", step_amp_strain_plot_file)