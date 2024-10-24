import os
import zipfile
import torch
from torchvision import transforms
from PIL import Image
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


zip_files = {
    'blur': 'blur.zip',
    'crop': 'cropped.zip',
    'dark': 'dark.zip',
    'contrast': 'contrast.zip',
    'normal': 'images.zip'
}
target_dir = 'data'

os.makedirs(target_dir, exist_ok=True)

for class_name, zip_path in zip_files.items():
    class_dir = os.path.join(target_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(class_dir)

print("Extraction complete!")

for class_name, zip_path in zip_files.items():
    if class_name == 'normal':
        images_dir = os.path.join(class_dir, 'images')
        if os.path.exists(images_dir) and os.path.isdir(images_dir):
            for item in os.listdir(images_dir):
                shutil.move(os.path.join(images_dir, item), class_dir)
            os.rmdir(images_dir)



num_classes = 5
batch_size = 32
num_epochs = 10
learning_rate = 0.001


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root='data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pt")  # Save the model's state dict
        print(f'Saved new best model with loss: {best_loss:.4f}')

torch.save(model.state_dict(), "model.pt")



