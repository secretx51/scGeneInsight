# Train.py to generate shap values etc nvid1021 lines total
from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from utilities.model_settings import FileSettings as FS, ModelTargets as MT
from utilities.hyperparameters import TuneHyperParameters as HP
from utilities.framework_setup import SetupML
from data.data_preprocessing import PreProcess_Anndata
from data.data_loader import FeatureLoader
from models.drop_connect import DropConnect
from models.multi_layer_MLP import DeepGeneExpressionClassifier
from training.train_model import ModelDimensions
from training.train_model import TuneTrainingLoop
from training.predict_model import Evaluation
from training.feature_importance import ShapImportance

class TuneModel():
    def __init__(self) -> None:
        self.model_setup = SetupML(FS.NUM_CPU, FS.MAX_RAM)
        
        self.adata
        self.inputDim = None
        self.outputDim = None
    
    def preProcess
    def load_data(self):

        features = FeatureLoader(adata=adata,
                                test_split=HP.TEST_SPLIT,
                                batch_size=HP.BATCH_SIZE,
                                num_workers=FS.NUM_CPU)
        return features.get_dataloaders()
    
    def setDims(self, adata):
        dims = ModelDimensions(adata, HP.QUANTILES)
        self.inputDim = dims.getInputDim()
        self.outputDim = dims.getOutputDim()

    def train_cifar(self):
        train_dataloader, test_dataloader, val_dataloader = self.load_data()

        dropconnect = DropConnect(HP.DROP_CONNECT)
        model = DeepGeneExpressionClassifier(input_size=self.inputDim, 
                                            hidden_sizes=HP.HIDDEN_DIM, 
                                            output_size=self.outputDim, 
                                            dropout_prob=HP.DROPOUT_RATE, 
                                            l1_reg=HP.L1_REG)
        
        model = self.model_setup.multi_gpu_model(model)
        device = self.model_setup.getDevice()

        train = TuneTrainingLoop(model, dropconnect, 
                            train_dataloader, val_dataloader, device, 
                            HP.LR, HP.WEIGHT_DECAY, HP.MOMENTUM, 
                            HP.MAX_LR, HP.PATIENCE, HP.NUM_EPOCHS)
        train.trainingLoop()

    def test_accruacy(self, model, test_dataloader, device):
        evaluate = Evaluation(model, test_dataloader, device, FS.FILE_DELIN)
        return evaluate.getAccuracy()

    def main(self, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
        self.load_data()
        config = HP().getConfig()
        
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2,
        )
        result = tune.run(
            partial(self.train_cifar),
            resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
        )

        best_trial = result.get_best_trial("loss", "min", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
        print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

        best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)
        

        best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                best_checkpoint_data = pickle.load(fp)

            best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
            test_acc = test_accuracy(best_trained_model, device)
            print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)