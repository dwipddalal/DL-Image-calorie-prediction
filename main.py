# Import required libraries
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F


# DataLoader
class FoodImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sub_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.image_paths = []

        for sub_dir in self.sub_dirs:
            sub_dir_path = os.path.join(self.root_dir, sub_dir)
            for filename in os.listdir(sub_dir_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.image_paths.append(os.path.join(sub_dir_path, filename))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = os.path.basename(os.path.dirname(img_path))

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data Preparation
root_dir = 'data/extracted_data/food_photos/'
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create dataset and dataloader
food_image_dataset = FoodImageDataset(root_dir=root_dir, transform=transform)

# Splitting the dataset into train and test sets
train_size = int(0.9 * len(food_image_dataset))
test_size = len(food_image_dataset) - train_size
train_dataset, test_dataset = random_split(food_image_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of testing examples: {len(test_dataset)}")

# Initialize and modify pre-trained ResNet model
resnet_model = models.resnet18(pretrained=True)
num_ftrs = resnet_model.fc.in_features
print(food_image_dataset.sub_dirs)
resnet_model.fc = nn.Linear(num_ftrs, len(food_image_dataset.sub_dirs))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=3e-4, momentum=0.9)

# Updated Fine-tuning function to save the best model based on test accuracy
def train(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, save_path='best_model.pth'):
    best_accuracy = 0.0 
    
    for epoch in range(num_epochs):
        model.train()
        
        train_epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            label_to_idx = {label: idx for idx, label in enumerate(train_loader.dataset.dataset.sub_dirs)}
            labels_idx = torch.tensor([label_to_idx[label] for label in labels]).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_idx)
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels_idx.size(0)
            train_correct += (predicted == labels_idx).sum().item()
        

        model.eval()
        
        test_epoch_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                
                
                label_to_idx = {label: idx for idx, label in enumerate(test_loader.dataset.dataset.sub_dirs)}
                labels_idx = torch.tensor([label_to_idx[label] for label in labels]).to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels_idx)
                
                test_epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels_idx.size(0)
                test_correct += (predicted == labels_idx).sum().item()
        
        # Calculate and print epoch loss and accuracy for both training and testing
        train_epoch_loss /= len(train_loader)
        train_epoch_accuracy = 100 * train_correct / train_total
        test_epoch_loss /= len(test_loader)
        test_epoch_accuracy = 100 * test_correct / test_total
        
        print(f"Epoch completed.\n Training Loss: {train_epoch_loss:.3f} | Training Accuracy: {train_epoch_accuracy:.2f}%")
        print(f" Testing Loss: {test_epoch_loss:.3f} | Testing Accuracy: {test_epoch_accuracy:.2f}%\n")
        
        # Save the best model
        if test_epoch_accuracy > best_accuracy:
            best_accuracy = test_epoch_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

# Uncomment the following line to actually run fine-tuning and save the best model
train(resnet_model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=100, save_path='best_model.pth')

## Inference
def inference(model, image_path, transform, class_to_idx):
    
    image = Image.open(image_path).convert("RGB")
    
    image_tensor = transform(image).unsqueeze(0)  
    image_tensor = image_tensor.to(device)

    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
        
    _, predicted_idx = torch.max(output.data, 1)
    predicted_idx = predicted_idx.item()
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_label = idx_to_class[predicted_idx]
    
    return predicted_label

# Load the best model
best_model_path = 'best_model.pth'
resnet_model.load_state_dict(torch.load(best_model_path))

# Prepare the class-to-index mapping
class_to_idx = {label: idx for idx, label in enumerate(food_image_dataset.sub_dirs)}

image_path1 = 'data/extracted_data/test_photos/VegBurger/burger1.jpeg'
image_path2 = 'data/extracted_data/test_photos/Pizza/pizza1.jpeg'
image_path3 = 'data/extracted_data/test_photos/FrenchFries/ffries1.jpeg'


predicted_label = inference(resnet_model, image_path2, transform, class_to_idx)
print(f"The predicted label for the image is: {predicted_label}")
