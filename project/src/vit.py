# Data organization for Vision Transformers (ViT) Base Model
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

# Declare Vision Transformers (ViT) Base Model
# Load pre-trained Vision Transformer (ViT)
vit_b = models.vit_b_16(weights="IMAGENET1K_V1")

# ViT classification head modified to output 5 classes
vit_b.heads.head = nn.Linear(vit_b.heads.head.in_features, 5, bias=True)

vit_b = vit_b.to(device)

vit_b.eval()

epochs = 50
learning_rate = 0.0001

criterion = nn.CrossEntropyLoss()
vit_optimizer = torch.optim.SGD(vit_b.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

train_model_with_acc(vit_b, criterion, vit_optimizer, epochs)

# Caculate Validation Accuracy, Matrix, and Metrics
confusion_matrix_and_metrics(vit_b, val_dataloader, device)
calculate_accuracy(vit_b, val_dataloader, device)

# Caculate Test Accuracy, Matrix, and Metrics
confusion_matrix_and_metrics(vit_b, test_dataloader, device)
calculate_accuracy(vit_b, test_dataloader, device)

