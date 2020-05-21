# =============================================================================
# Import all packages
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Create dataset class, load data, and batch it
# =============================================================================
class ProcessedCatDogData(Dataset):

    def __init__(self, csv_file_cat, csv_file_dog):
        self.df_cat = pd.read_csv(csv_file_cat, header=None)
        self.df_dog = pd.read_csv(csv_file_dog, header=None)
        self.df = pd.concat([self.df_cat, self.df_dog])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.df.iloc[idx, :-1].values
        y = self.df.iloc[idx, -1]

        return {"x": x, "y": y}

trainset = ProcessedCatDogData(csv_file_cat="cat_data.csv", csv_file_dog="dog_data.csv")

print("Number of data points in training set =", len(trainset))

trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)

print("Number of training batches =", len(trainloader))

# =============================================================================
# Create model class
# =============================================================================
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features=122, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x

# =============================================================================
# Instantiate model and set hyperparameters
# =============================================================================
num_epochs = 20
learning_rate = 1e-3

model = MLP()
distance = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# =============================================================================
# Train model
# =============================================================================
for epoch in range(num_epochs):
    for data in trainloader:
        x, y = data['x'], data['y']
        x = Variable(x).cpu()
        y = Variable(y).cpu()
        # ===================forward=====================
        output = model(x.float())
        loss = distance(output, y)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

# =============================================================================
# Save model
# =============================================================================
torch.save(model.state_dict(), "cat-dog-model.pth")
