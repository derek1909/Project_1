import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py
import ipdb
import yaml  # For logging training history to a YAML file

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
class Const_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        """
        Args:
            input_dim (int): Number of input features (ndim).
            hidden_dim (int): Size of the hidden layers.
            output_dim (int, optional): Number of output features.
                                        If None, set equal to input_dim.
        """
        super(Const_Net, self).__init__()
        if output_dim is None:
            output_dim = input_dim

        # Encoder: maps the input strain to a latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Decoder: maps the latent representation to predicted stress
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

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
        # Encode the flattened input
        encoded = self.encoder(x_flat)
        # Decode the latent representation
        decoded = self.decoder(encoded)
        # Reshape back to (batch_size, nstep, output_dim)
        out = decoded.view(batch_size, nstep, -1)
        return out

######################### Experiment Loop #########################
# List of material files (assumed to be in Problem_1_student/Data/)
# material_files = ['Material_A.mat', 'Material_B.mat', 'Material_C.mat']
material_files = ['Material_C.mat']

# Hyperparameters (you can experiment with these and comment in your report)
learning_rate = 3e-4
hidden_dim = 64
epochs = 200
batch_size = 20

# Base folders for data and results
data_folder = 'Problem_1_student/Data'
results_base = 'Problem_1_student/results'
os.makedirs(results_base, exist_ok=True)

for material_file in material_files:
    material_name = os.path.splitext(material_file)[0]  # e.g., "Material_A"
    print(f"Processing {material_name}...")

    # Create a folder for saving results for this material
    result_folder = os.path.join(results_base, material_name)
    os.makedirs(result_folder, exist_ok=True)

    # Read the data for the current material
    path = os.path.join(data_folder, material_file)
    data_reader = MatRead(path)
    strain = data_reader.get_strain()
    stress = data_reader.get_stress()

    # Shuffle and split into training and test sets (80/20 split)
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

    ndim = strain.shape[2]  # Number of components
    nstep = strain.shape[1] # Number of time steps
    dt = 1 / (nstep - 1)

    # Create DataLoader for training data
    train_set = Data.TensorDataset(train_strain_encode, train_stress_encode)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Initialize the network, loss function, optimizer, and scheduler
    net = Const_Net(input_dim=ndim, hidden_dim=hidden_dim, output_dim=ndim)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of parameters for {material_name}: {n_params}")

    loss_func = Lossfunc()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    loss_train_list = []
    loss_test_list = []

    # Training loop
    for epoch in range(epochs):
        net.train(True)
        trainloss = 0.0
        for i, data in enumerate(train_loader):
            inputs, targets = data
            output = net(inputs)
            loss = loss_func(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # update learning rate per batch

            trainloss += loss.item()
        trainloss /= len(train_loader)
        loss_train_list.append(trainloss)

        # Compute test loss
        net.eval()
        with torch.no_grad():
            output_test = net(test_strain_encode)
            testloss = loss_func(output_test, test_stress_encode).item()
        loss_test_list.append(testloss)

        if epoch % 10 == 0:
            print(f"{material_name} - epoch: {epoch}, train loss: {trainloss:.4f}, test loss: {testloss:.4f}")

    print(f"{material_name} - Final Train loss: {trainloss:.4f}")
    print(f"{material_name} - Final Test loss: {testloss:.4f}")

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

    # Plot and save train vs test loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), loss_train_list, label='Train Loss')
    plt.plot(range(epochs), loss_test_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{material_name}: Train vs Test Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_file = os.path.join(result_folder, 'train_vs_test_loss.png')
    plt.savefig(loss_plot_file)
    plt.close()
    print(f"Loss plot saved to {loss_plot_file}")

    # Plot one sample of truth stress vs predicted stress
    net.eval()
    with torch.no_grad():
        # Use the first sample from the test set
        sample_input = test_strain_encode[0:1]
        sample_target = test_stress_encode[0:1]
        sample_output = net(sample_input)
    time_steps = np.linspace(0, dt*(nstep-1), nstep)
    plt.figure(figsize=(8, 5))

    # Retrieve default color cycle from matplotlib
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for comp in range(ndim):
        color = colors[comp % len(colors)]
        # Plot truth stress in solid line
        plt.plot(time_steps, sample_target[0, :, comp].cpu().numpy(), 
                label=f'Truth Stress comp {comp}', color=color)
        # Plot predicted stress in dashed line using the same color
        plt.plot(time_steps, sample_output[0, :, comp].cpu().numpy(), '--', 
                label=f'Predicted Stress comp {comp}', color=color)

    plt.xlabel('Time')
    plt.ylabel('Stress')
    plt.title(f'{material_name}: Stress Comparison (Sample 0)')
    plt.legend()
    plt.grid(True)
    sample_plot_file = os.path.join(result_folder, 'stress_comparison_sample.png')
    plt.savefig(sample_plot_file)
    plt.close()
    print(f"Stress comparison plot saved to {sample_plot_file}")

print("All experiments completed.")