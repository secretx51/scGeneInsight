class FileSettings():
    # File settings
    H5_PATH = '/scratch/user/uqtneil1/GSE180661/all/pre-processing/processed_h5/GSE180661_final_anno.h5ad'
    GSVA_PATH = '/scratch/user/uqtneil1/GSE180661/all/aim2_ml/gene_sets/GSVA.csv'
    FILE_DELIN = 'rem32'
    # System Specs
    NUM_CPU = 32 # Number threads
    MAX_RAM = 384 # In GB
    # Boolean Params
    FILTER_KEY_GENES = True
    REM_NON_CODING = False
    MEMORY_LIMIT = False
    MULTI_GPU = False

class ModelTargets():
    CELL_TYPE = 'Ovarian.cancer.cell'
    KEY_GENES = ['B2M', 'IDE', 'HLA-G', 'MICA', 'MICB', 'HLA-H', 
                'ERAP2', 'ERAP1', 'HLA-B', 'TAP2', 'TAP1', 'HLA-C', 
                'HLA-A', 'HLA-F', 'HLA-E', 'TAPBP', 'RAET1E', 'RAET1G', 
                'RAET1L', 'HLA-DRA', 'ULBP2', 'ULBP1', 'HLA-DRB1', 'ULBP3', 'HFE']
    # KEY_GENES = ['B2M', 'CALR', 'NLRC5', 'CANX', 'ERAP1', 'ERAP2', 
    #             'HLA-A', 'HLA-B', 'HLA-C', 'PDIA3', 'PSMB5', 'PSMB6', 
    #             'PSMB7', 'PSMB8', 'PSMB9', 'PSMB10', 'PSME1', 'PSME3', 
    #             'RFX5', 'HSP90AB1', 'TAP1', 'TAP2', 'TAPBP']
    # CUSTOM_MHC_I trent_derived
    GENE_SET = 'ANTIGEN PROCESSING AND PRESENTATION OF ENDOGENOUS PEPTIDE ANTIGEN%GOBP%GO:0002483'   
