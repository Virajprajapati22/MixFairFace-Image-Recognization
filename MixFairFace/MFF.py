import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet34
from tqdm import tqdm

# Define your ResNet34 model
class ResNet34WithMid(nn.Module):
    def __init__(self):
        super(ResNet34WithMid, self).__init__()
        self.encoder = resnet34(pretrained=True)  # Load pre-trained ResNet34 model
        self.encoder.fc = nn.Identity()  # Replace the fully connected layer with an identity layer
        self.mid = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.mid(x)
        return x

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your face dataset
train_dataset = ImageFolder(root='/Users/viru/Documents/GitHub/MixFairFace-Image-Recognization/resources/data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize your model
model = ResNet34WithMid()
batch = []

# Define your custom loss function
class MixFairFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(MixFairFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        img_x = batch['rgb']
        class_x = batch['label']
        attribute_x = batch['attribute']
        
        ###
        feat = self.model.encoder(img_x)['l4']
        feat_a = F.normalize(self.model.mid(feat.flatten(1)))
        
        indices = torch.randperm(feat.shape[0])
        feat_b = feat_a[indices]
        feat_mix = 0.5 * feat + 0.5 * feat[indices]
        feat_mix = F.normalize(self.model.mid(feat_mix.flatten(1)))

        diff = ((feat_mix * feat_b).sum(-1, keepdim=True))**2 - ((feat_mix * feat_a).sum(-1, keepdim=True))**2
        pred = self.model.product(feat_a, class_x, diff)
        ####
        loss = nn.CrossEntropyLoss()(pred, class_x)

        out = {
                'loss': loss,
            }
        self.log('entropy-loss', loss, on_step=True)

        return out

# Get the number of classes from the train_data object
num_classes = len(train_dataset.classes)

custom_loss_function = MixFairFaceLoss(in_features=512, out_features=num_classes)  # Assuming 512 is the number of output features from your mid layer

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Train your model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        optimizer.zero_grad()
        
        # Forward pass
        features = model(images)
        
        # Calculate loss
        loss = custom_loss_function(features, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Optionally, you can adjust learning rate
    scheduler.step()

# Save your trained model
torch.save(model.state_dict(), '/Users/viru/Documents/GitHub/MixFairFace-Image-Recognization/models/trained_model.pth')
