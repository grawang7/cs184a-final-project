# Declare Data Efficient Image Transformer (DeiT)
import timm

deit = timm.create_model("deit_base_patch16_224", pretrained=True)
deit.head = nn.Linear(deit.head.in_features, 5, bias=True)
deit = deit.to(device)

epochs = 50
learning_rate = 0.0001

criterion = nn.CrossEntropyLoss()
deit_optimizer = torch.optim.Adam(deit.parameters(), lr=learning_rate)

train_model_with_acc(deit, criterion, deit_optimizer, epochs)

# Caculate Validation Accuracy, Matrix, and Metrics
confusion_matrix_and_metrics(deit, val_dataloader, device)
calculate_accuracy(deit, val_dataloader, device)

confusion_matrix_and_metrics(deit, test_dataloader, device)
calculate_accuracy(deit, test_dataloader, device)

