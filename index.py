import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet34
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load custom dataset
    trainset = ImageFolder(root='C:\\Users\\student\\Desktop\\D K\\Fairness-Image-Recognization\\MixFairFace\\trainingdata\\train', transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = ImageFolder(root='C:\\Users\\student\\Desktop\\D K\\Fairness-Image-Recognization\\MixFairFace\\trainingdata\\test', transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Define ResNet-34 model
    model = resnet34(pretrained=False, num_classes=len(trainset.classes)).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Training the model
    def train(model, trainloader, criterion, optimizer, epochs=1):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            pbar = tqdm(enumerate(trainloader), total=len(trainloader))
            for i, data in pbar:
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / (i+1):.4f}')

    # Evaluate the model
    def evaluate(model, testloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

    # Train the model
    train(model, trainloader, criterion, optimizer, epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), 'resnet34_model.pth')

    # Evaluate the model
    evaluate(model, testloader)
