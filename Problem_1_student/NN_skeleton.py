import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py
import ipdb
import yaml  # For logging training history to a YAML file

# Define your loss function here
class Lossfunc(object):
    def __init__(self):
        # Create an instance of the MSE loss function from PyTorch
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred_stress, target_stress):
        """
        Compute the MSE loss between predicted stress and target stress.
        
        Parameters:
            pred_stress (torch.Tensor): Predicted stress values.
            target_stress (torch.Tensor): Ground truth stress values.
        
        Returns:
            torch.Tensor: Computed MSE loss.
        """
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
        # Flatten the first two dimensions so that each row corresponds to a single sample at one time step.
        data_flat = data.reshape(-1, data.shape[-1])
        self.mean = data_flat.mean(axis=0)
        self.std = data_flat.std(axis=0)
        # Avoid division by zero in case a feature has zero variance.
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        """
        Standardize the data using the computed mean and std.
        Parameters:
            data (np.ndarray): Data array of shape (n_samples, n_steps, n_features)
        Returns:
            np.ndarray: Normalized data of the same shape as input.
        """
        if self.mean is None or self.std is None:
            raise ValueError("The normalizer must be fitted before calling transform.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data_normalized):
        """
        Revert the standardization to get the original data.
        Parameters:
            data_normalized (np.ndarray): Normalized data array.
        Returns:
            np.ndarray: Original data recovered from the normalized data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("The normalizer must be fitted before calling inverse_transform.")
        return data_normalized * self.std + self.mean

# Define network your neural network for the constitutive model below
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

######################### Data processing #############################
# Read data from .mat file
path = 'Problem_1_student/Data/Material_A.mat' #Define your data path here
data_reader = MatRead(path)
strain = data_reader.get_strain()
stress = data_reader.get_stress()

# Split data into train and test
n_samples = strain.shape[0]
ntrain = int(0.8 * n_samples)
ntest = n_samples - ntrain

# Apply the permutation
perm = torch.randperm(n_samples)
strain = strain[perm]
stress = stress[perm]

train_strain = strain[:ntrain]
train_stress = stress[:ntrain]
test_strain  = strain[ntrain:]
test_stress  = stress[ntrain:]

# Normalize your data
strain_normalizer   = DataNormalizer(train_strain)
train_strain_encode = strain_normalizer.transform(train_strain)
test_strain_encode  = strain_normalizer.transform(test_strain)

stress_normalizer   = DataNormalizer(train_stress)
train_stress_encode = stress_normalizer.transform(train_stress)
test_stress_encode  = stress_normalizer.transform(test_stress)

ndim = strain.shape[2]  # Number of components
nstep = strain.shape[1] # Number of time steps
dt = 1/(nstep-1)

# Create data loader
batch_size = 20
train_set = Data.TensorDataset(train_strain_encode, train_stress_encode)
train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

############################# Define and train network #############################
# Create Neural network, define loss function and optimizer
net = Const_Net(input_dim=ndim, hidden_dim=64, output_dim=ndim)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) #Calculate the number of training parameters
print('Number of parameters: %d' % n_params)

loss_func = Lossfunc() # define loss function
optimizer = torch.optim.Adam(net.parameters(), lr=3e-2)  # define optimizer with learning rate 0.001
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)  # decrease LR by half every 20 epochs

# Train network
epochs = 200  # define number of training epochs
print("Start training for {} epochs...".format(epochs))

loss_train_list = []
loss_test_list = []

for epoch in range(epochs):
    net.train(True)

    trainloss = 0.0
    for i, data in enumerate(train_loader):
        input, target = data
        # Forward pass
        output = net(input)
        loss = loss_func(output, target)

        # Clear gradients, backpropagate, and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # update learning rate if required per batch

        # Accumulate the training loss
        trainloss += loss.item()
    # ipdb.set_trace()
    trainloss /= len(train_loader)

    # Compute test loss using the normalized test data
    net.eval()
    with torch.no_grad():
        test_input = test_strain_encode  # normalized test strain data
        test_target = test_stress_encode  # normalized test stress data
        output_test = net(test_input)
        testloss = loss_func(output_test, test_target).item()

    # Print train loss every 10 epochs
    if epoch % 10 == 0:
        print("epoch: {}, train loss: {:.4f}, test loss: {:.4f}".format(epoch,
                                                                         trainloss,
                                                                         testloss))
    # Save loss values for plotting and logging
    loss_train_list.append(trainloss)
    loss_test_list.append(testloss)

print("Final Train loss: {:.4f}".format(trainloss))
print("Final Test loss: {:.4f}".format(testloss))

############################# Log training history to a YAML file #############################
history = {
    'epochs': epochs,
    'train_loss': loss_train_list,
    'test_loss': loss_test_list
}

with open('./Problem_1_student/training_history.yaml', 'w') as f:
    yaml.dump(history, f)
print("Training history has been logged to training_history.yaml")

############################# Plot your results using Matplotlib #############################
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_train_list, label='Train Loss')
plt.plot(range(epochs), loss_test_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
plt.legend()
plt.grid(True)
plt.savefig('./Problem_1_student/train_vs_test_loss.png')  # Save the figure as a PNG file
plt.close()  # Close the figure to free up memory