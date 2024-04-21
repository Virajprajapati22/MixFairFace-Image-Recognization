import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
# from cosface import CosFaceLoss
from tqdm import tqdm


# Define CosFace loss
class CosFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # Reshape input tensor to (batch_size, in_features)
        input = input.view(input.size(0), -1)
        
        # Transpose weight matrix
        weight = F.normalize(self.weight)
        
        # Compute cosine similarity
        cosine = F.linear(F.normalize(input), weight)

        # Compute one-hot encoding for the labels
        one_hot = F.one_hot(label, self.out_features).float()

        # Compute output logits
        output = self.s * (cosine - one_hot * self.m)
        
        # Compute sum of loss across all elements
        loss = F.cross_entropy(output, label)
        
        return loss


# Load data from directory structure using ImageFolder
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = ImageFolder(root='/Users/viru/Documents/GitHub/MixFairFace-Image-Recognization/resources/data/train/', transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Initialize ResNet34 model
model = models.resnet34(pretrained=False)
num_ftrs = model.fc.in_features

# Remove the final classification layer
model = nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Get the number of classes from the train_data object
num_classes = len(train_data.classes)

# Initialize CosFace loss function
criterion = CosFaceLoss(in_features=num_ftrs, out_features=num_classes, s=64.0, m=0.35)

# Initialize SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Learning rate scheduler with step decay
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 18, 30, 34], gamma=0.1)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            # Pass the images through the model and reshape the output
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)

            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.update(1)
            pbar.set_postfix({'Loss': running_loss / ((pbar.n) * images.size(0))})
    scheduler.step()  # Update learning rate scheduler
    epoch_loss = running_loss / len(train_data)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), '/Users/viru/Documents/GitHub/MixFairFace-Image-Recognization/models/resnet34_cosface_model.pth')