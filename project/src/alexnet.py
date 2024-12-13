# Data organization for AlexNet model
# do we need consistent shuffle across different models? (will it affect model evaluation?)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

# Declare AlexNet model
# https://analyticsindiamag.com/ai-mysteries/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/
alexnet_SGD = torchvision.models.alexnet(pretrained=True)
alexnet_SGD.eval()

# Update classifiers of model
alexnet_SGD.classifier[4] = nn.Linear(4096,1024) # update second classifier of model to reduce number of nodes in dense layers of network
alexnet_SGD.classifier[6] = nn.Linear(1024,5) # update output layer of network to have 5 class labels
alexnet_SGD.eval()

alexnet_SGD = alexnet_SGD.to(device)

epochs = 10
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
alexnet_optimizer_SGD = torch.optim.SGD(alexnet_SGD.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

train_model_with_acc(alexnet_SGD, criterion, alexnet_optimizer_SGD, epochs) # Test with SGD optimizer

# Calculate Validation Accuracy, Matrix, and Metrics with SGD optimizer
confusion_matrix_and_metrics(alexnet_SGD, val_dataloader, device)
calculate_accuracy(alexnet_SGD, val_dataloader, device)

# Calculate Test Accuracy, Matrix, and Metrics with SGD optimizer
confusion_matrix_and_metrics(alexnet_SGD, test_dataloader, device)
calculate_accuracy(alexnet_SGD, test_dataloader, device)

# Declare another Alex Net Model to use Adam optimizer
alexnet_Adam = torchvision.models.alexnet(pretrained=True)
alexnet_Adam.eval()

# Update classifiers of model
alexnet_Adam.classifier[4] = nn.Linear(4096,1024) # update second classifier of model to reduce number of nodes in dense layers of network
alexnet_Adam.classifier[6] = nn.Linear(1024,5) # update output layer of network to have 5 class labels
alexnet_Adam.eval()

alexnet_Adam = alexnet_Adam.to(device)

epochs = 20
learning_rate = 0.00001

criterion = nn.CrossEntropyLoss()
alexnet_optimizer_Adam = torch.optim.Adam(alexnet_Adam.parameters(), lr = learning_rate)
train_model_with_acc(alexnet_Adam, criterion, alexnet_optimizer_Adam, epochs) # Test with Adam optimizer

# Calculate Validation Accuracy, Matrix, and Metrics with Adam optimizer
confusion_matrix_and_metrics(alexnet_Adam, val_dataloader, device)
calculate_accuracy(alexnet_Adam, val_dataloader, device)

# Calculate Test Accuracy, Matrix, and Metrics with SGD optimizer
confusion_matrix_and_metrics(alexnet_Adam, test_dataloader, device)
calculate_accuracy(alexnet_Adam, test_dataloader, device)

