import torch
from torch.nn import DataParallel
import scanpy as sc

class SetupML():    
    def __init__(self, 
                threads: int, 
                memory: int) -> None:
        
        self.threads = threads
        self.memory = memory

        self.device = None
        self._setSettings()

    def scanpySettings(self):
        # Scanpy Settings
        sc.settings.verbosity = 3
        sc.settings.autosave = True
        sc.settings.file_format_figs = "png"
        sc.settings.n_jobs = self.threads
        sc.settings.max_memory = self.memory
        
    def torchSettings(self):
        # Torch settings
        torch.set_num_threads(self.threads)
        torch.set_float32_matmul_precision("high") # Enable tensor cores 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
    def _setSettings(self):
        self.scanpySettings()
        self.torchSettings()
    
    def getDevice(self):
        return self.device

    def multi_gpu_model(self, model):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = DataParallel(model)
        self.device = device
        return model