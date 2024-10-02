from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load training and validation datasets
train_data = datasets.ImageFolder('/path/to/data/train', transform=transform)
val_data = datasets.ImageFolder('/path/to/data/val', transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
