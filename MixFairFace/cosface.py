import torch
import torch.nn as nn

# Define CosFace loss
class CosFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# # Training loop
# num_epochs = 40
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.cuda(), labels.cuda() if torch.cuda.is_available() else (images, labels)
        
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item() * images.size(0)
    
#     scheduler.step()  # Update learning rate scheduler
#     epoch_loss = running_loss / len(train_data)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# # Save trained model
# torch.save(model.state_dict(), '/Users/viru/Documents/GitHub/MixFairFace-Image-Recognization/models/resnet34_cosface_model.pth')