class FileSettings():
    # File settings
    H5_PATH = '/scratch/user/uqtneil1/GSE180661/all/pre-processing/processed_h5/GSE180661_final_anno.h5ad'
    GSVA_PATH = '/scratch/user/uqtneil1/GSE180661/all/aim2_ml/gene_sets/GSVA.csv'
    FILE_DELIN = 'rem32'
    # System Specs
    NUM_CPU = 32 # Number threads
    MAX_RAM = 256 # In GB
    # Boolean Params
    FILTER_KEY_GENES = True
    REM_NON_CODING = True
    MEMORY_LIMIT = True
    MULTI_GPU = True

class ModelTargets():
    CELL_TYPE = 'Ovarian.cancer.cell'
    KEY_GENES = ['B2M', 'CALR', 'NLRC5', 'CANX', 'ERAP1', 'ERAP2', 
                'HLA-A', 'HLA-B', 'HLA-C', 'PDIA3', 'PSMB5', 'PSMB6', 
                'PSMB7', 'PSMB8', 'PSMB9', 'PSMB10', 'PSME1', 'PSME3', 
                'RFX5', 'HSP90AB1', 'TAP1', 'TAP2', 'TAPBP']
    GENE_SET = 'CUSTOM_MHC_I trent_derived'

class HyperParameters():
    # Hyperparameters
    # TODO: Tune, specifically try single layer hidden dim
    NUM_GENES = 1000
    TEST_SPLIT = 0.80
    BATCH_SIZE = 96 # set 96
    HIDDEN_DIM = [1024, 512]
    DROPOUT_RATE = 0.3
    DROP_CONNECT = 0.2
    L1_REG = 0.005
    LR = 0.001
    MAX_LR=0.1
    WEIGHT_DECAY = 0
    NUM_EPOCHS = 1000
    NOISE_AMOUNT = 0.4
    PATIENCE = 50
    MOMENTUM = 0.9
    QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]