import torch
from torchvision import transforms
from PIL import Image


model = torch.load("model.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict_image(model, image_path, device, class_names):
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    predicted_class = class_names[predicted.item()]
    print(f'Predicted Class: {predicted_class}')

class_names = ["blur", "contrast", "crop", "dark", "normal"]
image_path = '1.jpg'

predict_image(model, image_path, device, class_names)  #что-то такое можно вставлять в код приложения
