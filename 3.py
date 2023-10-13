import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torchinfo import summary

def load_images_from_folder(folderName):
    filenames = os.listdir(folderName)
    imagesTensor = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for filename in filenames:
        filePath = os.path.join(folderName, filename)
        with Image.open(filePath) as img:
            img = transform(img)
            imagesTensor.append(img)

    return torch.stack(imagesTensor)

class ModifiedMobileNet(nn.Module):
    def __init__(self):
        super(ModifiedMobileNet, self).__init__()
        mobilenet = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=None)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 2),   # 1280 matches the number of channels from MobileNetV2 features
            nn.Softmax(dim=1)
        )
        
        # Freeze the MobileNet layers
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def train_model(model, epochs=10, batchSize=32):
    # Load images and labels
    gliomaImages = load_images_from_folder("img/pizza")
    notumorImages = load_images_from_folder("img/steak")
    gliomaLabels = torch.tensor([[1, 0]] * len(gliomaImages))
    notumorLabels = torch.tensor([[0, 1]] * len(notumorImages))
    images = torch.cat((gliomaImages, notumorImages), 0)
    labels = torch.cat((gliomaLabels, notumorLabels), 0)

    # Split data into train and test sets
    split_idx = int(0.8 * len(images))  # Let's assume 80% of the data is for training
    train_images, test_images = images[:split_idx], images[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for data, target in train_dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, torch.argmax(target, dim=1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(torch.argmax(target, dim=1)).sum().item()

        # Testing
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                outputs = model(data)
                loss = criterion(outputs, torch.argmax(target, dim=1))
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += target.size(0)
                correct_test += predicted.eq(torch.argmax(target, dim=1)).sum().item()

        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss/len(train_dataloader):.4f}, '
              f'Train Accuracy: {100.*correct_train/total_train:.2f}%, '
              f'Test Loss: {test_loss/len(test_dataloader):.4f}, '
              f'Test Accuracy: {100.*correct_test/total_test:.2f}%')

        model.train()


def call():
    model = ModifiedMobileNet()
    summary(model, input_size=(1, 3, 224, 224))
    train_model(model, 10, 32)
    print("Model trained successfully")
    torch.save(model.state_dict(), "./saved.pth")

call()
