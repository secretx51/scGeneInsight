import torch
from aim2_ml.frame_work.utilities.training_setup import TrainingSetup

class TrainingLoop(TrainingSetup):
    def __init__(self, 
                input_dim, 
                output_size, 
                hidden_dim, 
                dropout_rate, 
                l1_reg, 
                device, 
                lr, 
                weight_decay, 
                momentum, 
                max_lr, 
                patience,
                num_epochs,
                train_dataloader,
                val_dataloader) -> None:
        super().__init__(input_dim, output_size, hidden_dim, dropout_rate, l1_reg, device, lr, weight_decay, momentum, max_lr)
        
        self.num_epochs = num_epochs
        self.patience = patience  # Number of epochs to wait for an improvement
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.best_val_loss = float('inf')  # Initialize with a large value
        self.stop_count = 0  # Counter for early stopping
        self.train_losses, self.val_losses, self.models, self.lrs = [], [], [], []

    # Add Gaussian noise
    def add_noise(self, inputs):
        noise = torch.randn(inputs.shape).to(self.device) * 0.1
        noisy_inputs = inputs + noise
        return noisy_inputs
    
    def resetTrain(self):
        self.train_loss = 0
        self.model.train()
    
    def forwardTrain(self, inputs, labels):
        with torch.autocast(device_type=self.device):
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels)
        self.model.l1_regularization() # L1 regularization
        return loss
    
    def updateScaler(self, loss, optimizer):
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
    def train(self):
        for inputs, labels in self.train_dataloader:
            self.optimizer.zero_grad()
            # Add Gaussian noise to avoid overfitting.
            inputs = self.add_noise(inputs)
            labels = labels.view(-1).long()
            loss = self.forwardTrain(inputs, labels)
            self.updateScaler(loss, self.optimizer)
            self.train_loss += loss.item()
            
    def trainMetrics(self):
        # Update learning rate
        self.train_losses.append(self.train_loss/len(self.train_dataloader))
        self.lrs.append(self.optimizer.param_groups[0]["lr"])
        self.scheduler.step()
        self.models.append(self.model)
        
    def resetVal(self)
        # Reset values for validation loss
        self.val_loss, self.accuracy = 0, 0

    def forwardValidate(self, inputs, labels):
        with torch.autocast(device_type='cuda'):
            outputs = self.model(inputs.float())
            loss = self.criterion(outputs, labels)
        return outputs, loss
    
    def updateValScores(self, labels, outputs, loss):
        self.val_loss += loss.item()
        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs, 1)
        self.accuracy += (predicted == labels).sum().item()
    
    def validate(self):
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                # inputs = inputs.view(inputs.shape[0], -1)  # Reshape inputs
                labels = labels.view(-1).long()  # Reshape labels to match the dimensions of outputs
                outputs, loss = self.forwardValidate(inputs, labels)
                self.updateValScores(labels, outputs, loss)
                
    def valMetrics(self, epoch):
        self.val_losses.append(self.val_loss/len(self.val_dataloader))
        
        print(f'Epoch: {epoch}/{self.num_epochs}',
            f'| Training loss: {self.train_loss/len(self.train_dataloader):.3e}',
            f'| Validation loss: {self.val_loss/len(self.val_dataloader):.4f}',
            f'| Validation accuracy: {self.accuracy/len(self.val_dataloader):.4f}')
        
    def earlyStop(self):
        if self.val_loss < self.best_val_loss:
            self.best_val_loss = self.val_loss
            self.stop_count = 0
        else:
            self.stop_count += 1
            if self.stop_count >= self.patience:
                print("Early stopping. Validation loss is increasing.")
                return True
        return False
        
    def trainingLoop(self):
        for epoch in range(1, self.num_epochs+1):
            self.resetTrain()
            self.train()
            self.resetTrain()
            self.validate()
            self.valMetrics(epoch)
            if self.earlyStop():
                break
    
    def getTrainLosses(self):
        return self.train_losses
    
    def getValLosses(self):
        return self.val_losses
    
    def getModels(self):
        return self.models
    
    def getBestModel(self):
        return self.models[self.val_losses.index(min(self.val_losses))]
    
    def getLrs(self):
        return self.lrs
