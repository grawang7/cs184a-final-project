# Data organization for ResNet Models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loads each subset directory from your "dataset" folder
train_dataset = datasets.ImageFolder(root='dataset/Train', transform=transform)
test_dataset = datasets.ImageFolder(root='dataset/Test', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/Val', transform=transform)

# Hyper Parameters for dataset organization and processing
batch_size = 25
num_workers = 3
# Dataloader for each subset which will batch, shuffle, and parallel load the data
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Declare ResNet50 Model
#weights=torchvision.models.ResNet50_Weights.DEFAULT
resnet_50 = torchvision.models.resnet50(pretrained=True)

for param in resnet_50.parameters():
    param.requires_grad = False
        
resnet_50.fc = nn.Linear(resnet_50.fc.in_features, 5)
resnet_50 = resnet_50.to(device)

epochs = 50
learning_rate = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet_50.parameters(), lr = learning_rate)

train_model(resnet_50, criterion, optimizer, epochs)

# Caculate Validation Accuracy, Matrix, and Metrics
confusion_matrix_and_metrics(resnet_50, val_dataloader, device)
calculate_accuracy(resnet_50, val_dataloader, device)

# Caculate Test Accuracy, Matrix, and Metrics
confusion_matrix_and_metrics(resnet_50, test_dataloader, device)
calculate_accuracy(resnet_50, test_dataloader, device)

