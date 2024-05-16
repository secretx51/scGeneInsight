from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.cuda.amp import GradScaler

from data.data_loader import DataLoader
from models.drop_connect import DropConnect
from models.multi_layer_MLP import DeepGeneExpressionClassifier

class ModelDimensions():
    def __init__(self, adata, le_dict) -> None:
        self.adata = adata
        self.le_dict = le_dict

    def getInputDim(self):
        return self.adata.var.shape[0]

    def getOutputDim(self):
        return len(self.le_dict)

class TrainingSetup():
    def __init__(self, 
                input_dim, 
                output_size, 
                hidden_dim, 
                dropout_rate, 
                l1_reg, device, 
                lr, weight_decay, 
                momentum, 
                max_lr) -> None:
    
        # Model Hyperparemeters
        self.input_dim = input_dim
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_lr = max_lr
        
        # Create training items
        self.model = self._createModel()
        self._getDropConnectLayers(self.model)
        self.criterion = self._createCriterion()
        self.optimizer = self._createOptimizer()
        self.scheduler = self._createScheduler()
        self.scaler = self._createScaler()
        
    def _createModel(self):
        return DeepGeneExpressionClassifier(self.input_dim, 
                                            self.hidden_dim, 
                                            self.output_size, 
                                            self.dropout_rate, 
                                            self.l1_reg).to(self.device)
    
    def _getDropConnectLayers(self, model):
        return [layer for layer in model.children() if isinstance(layer, DropConnect)]
    
    def getModel(self):
        return self.model

    def _createCriterion(self):
        return CrossEntropyLoss()

    def getCriterion(self):
        return self.criterion

    def _createOptimizer(self):
        return SGD(self.model.parameters(), 
                    lr=self.lr,
                    weight_decay=self.weight_decay, 
                    momentum=self.momentum)

    def getOptimizer(self):
        return self.optimizer

    def _createScheduler(self):
        return CyclicLR(self.optimizer, 
                        base_lr=self.lr, 
                        max_lr=self.max_lr, 
                        step_size_up=5, 
                        mode="triangular2")

    def getScheduler(self):
        return self.scheduler

    def _createScaler(self):
        return GradScaler()

    def getScaler(self):
        return self.scaler
