import torch
import scanpy as sc

from data.data_preprocessing import PreProcess_Anndata
from data.data_loader import DataLoader
from aim2_ml.frame_work.utilities.training_setup import ModelDimensions
from training.training_loop import TrainingLoop
from training.evaluation import Evaluation
from training.feature_importance import ShapImportance



def preprocessAdata():
    
    PreProcess_Anndata(
        cell_type='B cell',
        key_genes=['CD79A', 'CD79B', 'CD19'],
        filter_key='ES',
        gene_set='GO_B_CELL_ACTIVATION',
        h5_path=S,
        gsea_path='data/gsea.csv',
        file_delin=',',
        quantiles=[0.25, 0.5, 0.75],
        threads=8,
        memory=8,
        rem_non_coding=True,
        memory_limit=False,
        multi_gpu=False
    ).run()