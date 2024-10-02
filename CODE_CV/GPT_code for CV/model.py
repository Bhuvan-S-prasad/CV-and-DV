import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  

num_epochs = 10  

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

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

   
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total}%')
