# Define a train_model function
def train_model(model, criterion, optimizer, epochs):
    total_step = len(train_dataloader)
    
    for epoch in range(epochs):
        loss_per_epoch = 0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_per_epoch += loss.item()
            #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    #.format(epoch+1, epochs, i+1, total_step, loss.item()))
        print(f"Epoch: {epoch+1}, {loss_per_epoch/total_step}")

# train_model function with accuracy calculation to determine overfitting point
def train_model_with_acc(model, criterion, optimizer, epochs):
    total_step = len(train_dataloader)
    
    for epoch in range(epochs):
        loss_per_epoch = 0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_per_epoch += loss.item()
            #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    #.format(epoch+1, epochs, i+1, total_step, loss.item()))
        print(f"Epoch: {epoch+1}, {loss_per_epoch/total_step}")
        calculate_accuracy(model, val_dataloader, device)

