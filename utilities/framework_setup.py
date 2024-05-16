import torch
import scanpy as sc
from typing import List

class SetupML():
    # File settings
    H5_PATH = '/scratch/user/uqtneil1/GSE180661/all/pre-processing/processed_h5/GSE180661_final_anno.h5ad'
    GSVA_PATH = '/scratch/user/uqtneil1/GSE180661/all/aim2_ml/gene_sets/GSVA.csv'
    FILE_DELIN = 'rem32'
    REM_NON_CODING = True
    MEMORY_LIMIT = False
    MULTI_GPU = True
    # System Specs
    NUM_CPU = 32 # Number threads
    MAX_RAM = 256 # In GB
    
    def __init__(self, 
                h5_path: str, 
                gsea_path: str, 
                file_delin: str, 
                quantiles: List[float],
                threads: int, 
                memory: int,
                rem_non_coding: bool,
                memory_limit: bool, 
                multi_gpu: bool) -> None:
        
        self.h5_path = h5_path
        self.gsea_path = gsea_path
        self.file_delin = file_delin
        self.quantiles = quantiles
        self.threads = threads
        self.memory = memory
        self.rem_non_coding = rem_non_coding
        self.memory_limit = memory_limit
        self.multi_gpu = multi_gpu
        
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
    
    def getH5Path(self):
        return self.h5_path
    
    def getGseaPath(self):
        return self.gsea_path
    
    def getFileDelin(self):
        return self.file_delin
    
    def getDevice(self):
        return self.device
