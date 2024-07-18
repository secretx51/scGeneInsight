# Train.py to generate shap values etc nvid1021 lines total
import os
import tempfile
from functools import partial
from pathlib import Path

import ray.cloudpickle as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from ray import tune, train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler

from data.data_loader import FeatureLoader
from data.data_preprocessing import PreProcess_Anndata
from models.drop_connect import DropConnect
from models.multi_layer_MLP import DeepGeneExpressionClassifier
from training.feature_importance import ShapImportance
from training.predict_model import Evaluation
from training.train_model import ModelDimensions, TrainingLoop, TuneTrainingLoop
from utilities.framework_setup import SetupML
from utilities.hyperparameters import HyperParameters as HP, TuneHyperParameters as THP
from utilities.model_settings import FileSettings as FS, ModelTargets as MT
from visualisation.lr_plot import PlotLRS

class runModel():
    def __init__(self,
                test_split = HP.TEST_SPLIT,
                batch_size = HP.BATCH_SIZE,
                hidden_dim = HP.HIDDEN_DIM,
                dropout_rate = HP.DROPOUT_RATE,
                l1_reg = HP.L1_REG,
                lr = HP.LR,
                max_lr = HP.MAX_LR,
                weight_decay = HP.WEIGHT_DECAY,
                noise_amount = HP.NOISE_AMOUNT,
                momentum = HP.MOMENTUM,
                ) -> None:
        self.test_split = test_split
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.lr = lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.noise_amount = noise_amount
        self.momentum = momentum
        
        self.model_setup = SetupML(FS.NUM_CPU, FS.MAX_RAM)
        self.adata = self.preprocessAdata()
        self.le_dict = None
        self.shap_data = None
        self.model = None


    def preprocessAdata(self):
        preprocess = PreProcess_Anndata(MT.CELL_TYPE, MT.KEY_GENES, MT.GENE_SET, HP.QUANTILES, 
                                    FS.H5_PATH, FS.GSVA_PATH, FS.FILE_DELIN, 
                                    FS.FILTER_KEY_GENES, FS.REM_NON_CODING, FS.MEMORY_LIMIT)
        adata = preprocess.preprocess_adata()
        self.le_dict = preprocess.get_le_dict() # Convert int labels to quantile
        return adata
    
    def featureExtraction(self, adata):
        features = FeatureLoader(adata=adata,
                            test_split=self.test_split,
                            batch_size=self.batch_size,
                            num_workers=FS.NUM_CPU)
        self.shap_data = features.get_shap_data()
        return features.get_dataloaders()
    
    def createModel(self, adata):
        dims = ModelDimensions(adata, HP.QUANTILES)
        dropconnect = DropConnect(HP.DROP_CONNECT)
        self.model = DeepGeneExpressionClassifier(input_size=dims.getInputDim(), 
                                    hidden_sizes=self.hidden_dim, 
                                    output_size=dims.getOutputDim(), 
                                    dropout_prob=self.dropout_rate, 
                                    l1_reg=self.l1_reg)
        return dropconnect
    
    def checkMultiGPU(self):
        if FS.MULTI_GPU:
            self.model = self.model_setup.multi_gpu_model(self.model)
        device = self.model_setup.getDevice()
        return device
    
    def trainModel(self, dropconnect, train_dataloader, val_dataloader, device):
        train = TrainingLoop(self.model, dropconnect, 
                            train_dataloader, val_dataloader, device, 
                            self.lr, self.weight_decay, self.momentum, 
                            self.max_lr, HP.PATIENCE, HP.NUM_EPOCHS)
        train.trainingLoop() # Start model training
        PlotLRS(train.getTrainLosses(), train.getValLosses(), FS.FILE_DELIN).plot()
        return train.getBestModel()
    
    def evaluateModel(self, test_dataloader, device):
        evaluate = Evaluation(self.model, test_dataloader, device, FS.FILE_DELIN) # Runs on init
        evaluate.displayMetrics()
        evaluate.saveModel()
    
    def runModel(self):
        train_dataloader, test_dataloader, val_dataloader = self.featureExtraction(self.adata)
        # Creates Model
        dropconnect = self.createModel(self.adata)
        device = self.checkMultiGPU()
        self.model = self.trainModel(dropconnect, train_dataloader, val_dataloader, device)
        self.evaluateModel(test_dataloader, device)

    def featureImportance(self):
        # Runs on init
        ShapImportance(self.adata, self.shap_data, self.model, FS.FILE_DELIN)

class TuneModel(runModel):
    def __init__(self, 
                test_split=THP.TEST_SPLIT, 
                batch_size=THP.BATCH_SIZE, 
                hidden_dim=THP.HIDDEN_DIM, 
                dropout_rate=THP.DROPOUT_RATE, 
                l1_reg=THP.L1_REG, lr=THP.LR, 
                max_lr=THP.MAX_LR, 
                weight_decay=THP.WEIGHT_DECAY, 
                noise_amount=THP.NOISE_AMOUNT, 
                momentum=THP.MOMENTUM) -> None:
        super().__init__(test_split, batch_size, hidden_dim, dropout_rate, l1_reg, lr, max_lr, weight_decay, noise_amount, momentum)
        self.test_dataloader = None
        self.device = None
    
    # override
    def trainModel(self, dropconnect, train_dataloader, val_dataloader, device):
        return TuneTrainingLoop(self.model, dropconnect, 
                            train_dataloader, val_dataloader, device, 
                            self.lr, self.weight_decay, self.momentum, 
                            self.max_lr, HP.PATIENCE, HP.NUM_EPOCHS)
    
    def runModel(self):
        train_dataloader, self.test_dataloader, val_dataloader = self.featureExtraction(self.adata)
        # Creates Model
        dropconnect = self.createModel(self.adata)
        self.device = self.checkMultiGPU()
        train = self.trainModel(dropconnect, train_dataloader, val_dataloader, self.device)
        train.trainingLoop()
        
    # override
    def evaluateModel(self):
        if self.test_dataloader is None or self.device is None:
            raise ValueError("Must use the runModel method before evaluating.")
        evaluate = Evaluation(self.model, self.test_dataloader, self.device, FS.FILE_DELIN)
        return evaluate.getAccuracy()

def runFeatureImportance():
    print('Model Initiated; Processing data')
    model = runModel() # init
    print('Begin training and evaluate model')
    model.runModel() # trains/evaluates
    print('Running feature importance')
    model.featureImportance() # feature importance

if __name__ == "__main__":
    runFeatureImportance()
