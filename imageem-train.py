# =============================================================================
# Import all packages
# =============================================================================
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import argparse

# =============================================================================
# Instantiate command line argument parser
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--training", help="Path to training dataset", default="../demos/cat-and-dog/dataset/training_set")
parser.add_argument("--testing", help="Path to testing dataset", default="../demos/cat-and-dog/dataset/test_set")
parser.add_argument("--imageem", help="Name of file to save imageem embeddings in")
args = parser.parse_args()

# =============================================================================
# Load dataset, transform and batch it
# =============================================================================
transform = transforms.Compose([
    transforms.Scale((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247,0.243, 0.261))
])

trainset = tv.datasets.ImageFolder(root=args.training, transform=transform)
testset = tv.datasets.ImageFolder(root=args.testing, transform=transform)

print("Training set size =", len(trainset))
print("Testing set size =", len(testset))

BATCH_SIZE = 32

dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print("Training batches =", len(dataloader))
print("Testing batches =", len(testloader))

# =============================================================================
# Define the autoencoder architecture
# =============================================================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2)

        self.linear1 = nn.Linear(in_features=128*30*30,out_features=100)
        self.linear2 = nn.Linear(in_features=100,out_features=128*30*30)

        self.conv1_transpose = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,output_padding=1)
        self.conv2_transpose = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,output_padding=1)
        self.conv3_transpose = nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=3,stride=1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),128*30*30)
        imageem_vector = F.relu(self.linear1(x))
        x = F.relu(self.linear2(latent_vector))
        x = x.view(x.size(0),128,30,30)
        x = F.relu(self.conv1_transpose(x))
        x = F.relu(self.conv2_transpose(x))
        x = F.relu(self.conv3_transpose(x))

        return x,imageem_vector

# =============================================================================
# Instantiate model and set model hyperparameters
# =============================================================================
num_epochs = 50
learning_rate = 1e-5

model = Autoencoder()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# =============================================================================
# Training loop
# =============================================================================
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cpu()
        # ===================forward=====================
        output, _ = model(img)
        loss = distance(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

# =============================================================================
# Save the imageem vector
# =============================================================================
imageem_vectors = []
for i, (images, _) in enumerate(testloader):
    if(i == 2):
        break
    _, imageem = model(images)
    imageem = torch.mean(imageem, 0)
    imageem_vectors.append(imageem.detach().numpy())
imageem_vectors = np.array(imageem_vectors)
imageem_vectors = np.mean(imageem_vectors, axis=0)
print("ImageEm vector shape: ", latent_vectors.shape)

np.save("npy/" + args.imageem + ".npy")
np.savetxt("txt/" + args.imageem + ".txt")
