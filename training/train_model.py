import tempfile
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.cuda.amp import GradScaler
from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle

class ModelDimensions():
    def __init__(self, adata, quantiles) -> None:
        self.adata = adata
        self.quantiles = quantiles

    def getInputDim(self):
        return self.adata.var.shape[0]

    def getOutputDim(self):
        return len(self.quantiles) + 1 # For the greater than quantile

class TrainingSetup():
    def __init__(self, 
                model, 
                dropconnect,
                device, 
                lr, 
                weight_decay, 
                momentum, 
                max_lr) -> None:
    
        # Model Hyperparemeters
        self.model = model
        self.dropconnect = dropconnect
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_lr = max_lr
        
        # Create training items
        self.model = self._toDevice()
        self._getDropConnectLayers(self.model)
        self.criterion = self._createCriterion()
        self.optimizer = self._createOptimizer()
        self.scheduler = self._createScheduler()
        self.scaler = self._createScaler()
        
    def _toDevice(self):
        return self.model.to(self.device)
    
    def _getDropConnectLayers(self, model):
        return [layer for layer in model.children() if isinstance(layer, type(self.dropconnect))]

    def _createCriterion(self):
        return CrossEntropyLoss()

    def _createOptimizer(self):
        return SGD(self.model.parameters(), 
                    lr=self.lr,
                    weight_decay=self.weight_decay, 
                    momentum=self.momentum)

    def _createScheduler(self):
        return CyclicLR(self.optimizer, 
                        base_lr=self.lr, 
                        max_lr=self.max_lr, 
                        step_size_up=5, 
                        mode="triangular2")

    def _createScaler(self):
        return GradScaler()

class TrainingLoop(TrainingSetup):
    def __init__(self, 
                model,
                dropconnect,
                train_dataloader,
                val_dataloader,
                device, 
                lr, 
                weight_decay, 
                momentum, 
                max_lr, 
                patience,
                num_epochs,
                ) -> None:
        super().__init__(model, dropconnect, device, lr, weight_decay, momentum, max_lr)
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.patience = patience  # Number of epochs to wait for an improvement

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
        with torch.autocast(device_type="cuda"):
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            # Add Gaussian noise to avoid overfitting.
            inputs = self.add_noise(inputs)
            # Reshape
            labels = labels.view(-1).long()
            # Forward pass
            loss = self.forwardTrain(inputs, labels)
            self.updateScaler(loss, self.optimizer)
            self.train_loss += loss.item()
            
    def trainMetrics(self):
        # Update learning rate
        self.train_losses.append(self.train_loss/len(self.train_dataloader))
        self.lrs.append(self.optimizer.param_groups[0]["lr"])
        self.models.append(self.model)
        
    def resetVal(self):
        # Reset values for validation loss
        self.val_loss, self.accuracy, self.val_count = 0, 0, 0
        self.model.eval()

    def forwardValidate(self, inputs, labels):
        with torch.autocast(device_type="cuda"):
            outputs = self.model(inputs.float())
            loss = self.criterion(outputs, labels)
        return outputs, loss
    
    def updateValScores(self, labels, outputs, loss):
        self.val_count += 1 # Keep track of dataloader count
        self.val_loss += loss.item()
        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs, 1)
        self.accuracy += (predicted == labels).sum().item()
    
    def validate(self):
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # inputs = inputs.view(inputs.shape[0], -1)  # Reshape inputs
                labels = labels.view(-1).long()  # Reshape labels to match the dimensions of outputs
                outputs, loss = self.forwardValidate(inputs, labels)
                self.updateValScores(labels, outputs, loss)
                
    def valMetrics(self):
        self.val_losses.append(self.val_loss/len(self.val_dataloader))
        training_loss = self.train_loss/len(self.train_dataloader)
        validation_loss = self.val_loss/len(self.val_count)
        validation_accuracy = self.accuracy/len(self.val_count)
        return training_loss, validation_loss, validation_accuracy
    
    def printMetrics(self, epoch, training_loss, validation_loss, validation_accuracy):
        print(f'Epoch: {epoch}/{self.num_epochs}',
            f'| Training loss: {training_loss:.3e}',
            f'| Validation loss: {validation_loss:.4f}',
            f'| Validation accuracy: {validation_accuracy:.4f}')
        
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
            self.trainMetrics()
            self.resetVal()
            self.validate()
            training_loss, validation_loss, validation_accuracy = self.valMetrics()
            self.printMetrics(epoch, training_loss, validation_loss, validation_accuracy)
            self.scheduler.step() # Update LR
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

class TuneTrainingLoop(TrainingLoop):
    def __init__(self, model, dropconnect, train_dataloader, val_dataloader, device, lr, weight_decay, momentum, max_lr, patience, num_epochs) -> None:
        super().__init__(model, dropconnect, train_dataloader, val_dataloader, device, lr, weight_decay, momentum, max_lr, patience, num_epochs)
        self.start_epoch = self.createCheckpoint()
    
    def createCheckpoint(self):
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
                self.model.load_state_dict(checkpoint_state["net_state_dict"])
                self.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        else:
            start_epoch = 0
        return start_epoch
    
    def checkpointCallback(self, epoch, validation_loss, validation_accuracy):
        checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": validation_loss, "accuracy": validation_accuracy},
                checkpoint=checkpoint,
            )
    
    def trainingLoop(self):
        for epoch in range(self.start_epoch, self.num_epochs+1):
            self.resetTrain()
            self.train()
            self.trainMetrics()
            self.resetVal()
            self.validate()
            training_loss, validation_loss, validation_accuracy = self.valMetrics()
            self.printMetrics(epoch, training_loss, validation_loss, validation_accuracy)
            self.checkpointCallback(epoch, validation_loss, validation_accuracy)
            self.scheduler.step() # Update LR
            if self.earlyStop():
                break
