import numpy as np
from ray import tune

class HyperParameters():
    TEST_SPLIT = 0.80
    BATCH_SIZE = 256 # set 96
    HIDDEN_DIM = [1024, 512]
    DROPOUT_RATE = 0.4
    DROP_CONNECT = 0.2
    L1_REG = 0.005
    LR = 0.001
    MAX_LR=0.1
    WEIGHT_DECAY = 0
    NUM_EPOCHS = 300
    NOISE_AMOUNT = 0.4
    PATIENCE = 50
    MOMENTUM = 0.9
    QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

class TuneHyperParameters(HyperParameters):
    def __init__(self) -> None:
        self.config = self._createConfig()
        self._addHiddenDims()
        
        self.TEST_SPLIT = self.config['test_split']
        self.BATCH_SIZE = self.config['batch_size']
        self.HIDDEN_DIM = self._compileHiddenDims()
        self.DROPOUT_RATE = self.config['dropout_rate']
        self.L1_REG = self.config['l1_reg']
        self.LR = self.config['lr']
        self.MAX_LR= self.config['max_lr']
        self.WEIGHT_DECAY = self.config['weight_decay']
        self.NOISE_AMOUNT = self.config['noise_amount']
        self.MOMENTUM = self.config['momentum']
    
    def _createConfig(self):
        config = {
        "test_split": tune.choice([i for i in np.arange(0.5, 1, 0.1)]),
        "batch_size": tune.choice([16, 32, 64, 128, 256]),
        "num_dims": tune.choice([i for i in range(1,5)]),
        "dropout_rate": tune.choice([i for i in np.arange(0, 1, 0.1)]),
        "l1_reg": tune.loguniform(1e-4, 1e-1),
        "lr": tune.loguniform(1e-5, 1e-3),
        "max_lr": tune.loguniform(1e-3, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "noise_amount": tune.choice([i for i in np.arange(0, 1, 0.1)]),
        "momentum": tune.choice([i for i in np.arange(0, 1.1, 0.1)])
        }
        return config
    
    def _addHiddenDims(self):
        for i in range(self.config['num_dims']):
            # Size 32 to 2048 in powers of 2 for the number of dims: 1-4
            self.config[f'l{i}'] = tune.choice([2**i for i in range(5, 12)])
            
    def _compileHiddenDims(self):
        return [self.config[f'l{i}'] for i in range(self.config['num_dims'])]

    def getConfig(self):
        return self.config