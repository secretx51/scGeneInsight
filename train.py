from utilities.hyperparameters import HyperParameters as HP, FileSettings as FS, ModelTargets as MT
from utilities.framework_setup import SetupML

from data.data_preprocessing import PreProcess_Anndata
from data.data_loader import FeatureLoader

from models.drop_connect import DropConnect
from models.multi_layer_MLP import DeepGeneExpressionClassifier

from training.train_model import ModelDimensions
from training.train_model import TrainingLoop
from training.predict_model import Evaluation
from training.feature_importance import ShapImportance

from visualisation.lr_plot import PlotLRS

model_setup = SetupML(FS.NUM_CPU, FS.MAX_RAM)
device = model_setup.getDevice()

preprocess = PreProcess_Anndata(MT.CELL_TYPE, MT.KEY_GENES, MT.GENE_SET, HP.QUANTILES, 
                            FS.H5_PATH, FS.GSVA_PATH, FS.FILE_DELIN, 
                            FS.FILTER_KEY_GENES, FS.REM_NON_CODING, FS.MEMORY_LIMIT)
adata = preprocess.preprocess_adata()
le_dict = preprocess.get_le_dict()

features = FeatureLoader(adata=adata,
                        device=device,
                        test_split=HP.TEST_SPLIT,
                        batch_size=HP.BATCH_SIZE,
                        num_workers=FS.NUM_CPU)
train_dataloader, test_dataloader, val_dataloader = features.get_dataloaders()
shap_data = features.get_shap_data()

dims = ModelDimensions(adata, le_dict)

dropconnect = DropConnect(HP.DROP_CONNECT)
model = DeepGeneExpressionClassifier(input_size=dims.getInputDim(), 
                                    hidden_sizes=HP.HIDDEN_DIM, 
                                    output_size=dims.getOutputDim(), 
                                    dropout_prob=HP.DROPOUT_RATE, 
                                    l1_reg=HP.L1_REG)

train = TrainingLoop(model, dropconnect, 
                    train_dataloader, val_dataloader, model_setup.getDevice(), 
                    HP.LR, HP.WEIGHT_DECAY, HP.MOMENTUM, 
                    HP.MAX_LR, HP.PATIENCE, HP.NUM_EPOCHS)
PlotLRS(train.getTrainLosses(), train.getValLosses(), FS.FILE_DELIN).plot()
model = train.getBestModel()

evaluate = Evaluation(model, test_dataloader, FS.FILE_DELIN)
evaluate.saveModel()

shap = ShapImportance(adata, shap_data, model, FS.FILE_DELIN)
