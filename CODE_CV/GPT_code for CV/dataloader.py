from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_data = datasets.ImageFolder('C:\Users\bhuva\Desktop\BIRDS DS', transform=transform)
val_data = datasets.ImageFolder('C:\Users\bhuva\Desktop\BIRDS DS', transform=transform)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
