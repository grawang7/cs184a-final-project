# Custom CNN Model

class CustomCNN(nn.Module):
    def __init__(self, dropout, kernel, channels):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=kernel, stride=1, padding=0), 
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)) 

        output_shape = output_size(128, kernel, 0, 1, 2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=kernel, stride=1, padding=0), 
            nn.BatchNorm2d(channels*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)) 

        output_shape = output_size(output_shape, kernel, 0, 1, 2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, kernel_size=kernel, stride=1, padding=0), 
            nn.BatchNorm2d(channels*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)) 

        output_shape = output_size(output_shape, kernel, 0, 1, 2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels*4, channels*2, kernel_size=kernel, stride=1, padding=0), 
            nn.BatchNorm2d(channels*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2))

        output_shape = output_size(output_shape, kernel, 0, 1, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=kernel, stride=1, padding=0), 
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2))

        output_shape = output_size(output_shape, kernel, 0, 1, 2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(
            nn.Linear(channels*int(output_shape)*int(output_shape), channels*2),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(channels*2, channels),
            nn.ReLU())
        
        self.fc3 = nn.Linear(channels, 5)

    def forward(self, out):
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

# Test run, should output torch.Size([batch_size, classes])
custom_cnn = CustomCNN(dropout=0.3, kernel=3, channels=32).to(device)
output = custom_cnn(images.to(device))
print(output.size())

# Declare Custom CNN Model
custom_cnn = CustomCNN(dropout=0.1, kernel=3, channels=32).to(device)

# Set Hyper Parameters
epochs = 50
learning_rate = 0.0001

# Declare Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_cnn.parameters(), lr = learning_rate)

# Train Custom CNN Model
train_model(custom_cnn, criterion, optimizer, epochs)

# Calculate Custom CNN Accuracy, Matrix, and Metrics
confusion_matrix_and_metrics(custom_cnn, val_dataloader, device)
calculate_accuracy(custom_cnn, val_dataloader, device)

# Implementation of Bayesian Optimization to improve Hyper Parameters

# Representation of how our model is changed
def objective(params):
    
    learning_rate = params[0][0]
    batch_size = int(params[0][1])
    dropout = params[0][2]
    kernel = int(params[0][3])
    channels = int(params[0][4])
    
    custom_cnn = CustomCNN(dropout, kernel, channels).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(custom_cnn.parameters(), lr = learning_rate)

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    custom_cnn.train()
    for epoch in range(20):
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = custom_cnn(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    custom_cnn.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = custom_cnn(images)
            _, predicted = torch.max(outputs, 1)
    
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    print(-accuracy)
    return -accuracy

# Range of tested values in Bayesian Optimization
domain = [
    {"name": "learning_rate", "type": "continuous", "domain": (1e-5, 1e-2)},
    {"name": "batch_size", "type": "discrete", "domain": [25, 50, 100, 200]},
    {"name": "dropout", "type": "continuous", "domain": (0.1, 0.7)},
    {"name": "kernel", "type": "discrete", "domain": [3, 5, 7]},
    {"name": "channels", "type": "discrete", "domain": [8, 16, 32, 64, 128]}
]

# Run Bayesian Optimization
opt = GPyOpt.methods.BayesianOptimization(f=objective, domain=domain)
opt.run_optimization(max_iter=10)

# Print Results
best_params = opt.X[np.argmin(opt.Y)]
param_names = ["Learning Rate", "Batch Size", "Dropout", "Kernel Size", "Channels"]

print("Best Hyper Parameters: ")
for param_name, best_param in zip(param_names, best_params):
    if param_name == "Learning Rate" or param_name == "Dropout":
        print(f'{param_name}: {float(best_param)}')
    else:
        print(f'{param_name}: {int(best_param)}')
print(f'Loss: {-np.min(opt.Y)}')

# Test Custom CNN Accuracy, Matrix, and Metrics
confusion_matrix_and_metrics(custom_cnn, test_dataloader, device)
calculate_accuracy(custom_cnn, test_dataloader, device)

