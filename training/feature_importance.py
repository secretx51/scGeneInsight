import pickle
import shap
import torch
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ShapImportance():
    def __init__(self, adata, X_test, model, FILE_DELIN) -> None:
        self.adata = adata
        self.X_test = X_test
        self.model = model
        self.FILE_DELIN = FILE_DELIN
        
    def getShapData(self, X_test):
        return X_test.cpu().numpy().astype(np.float32).copy()
    
    def getShapSelection(self, shap_data):
        # Generate 1000 unique random indices
        explain_idx = np.random.choice(shap_data.shape[0], 1000, replace=False)
        # Select 1000 random rows
        explain_data = shap_data[explain_idx, :]
        # Generate 100 unique random indices
        shap_idx = np.random.choice(shap_data.shape[0], 100, replace=False)
        # Select 100 random rows
        shap_data = shap_data[shap_idx, :]
        return explain_data, shap_data

    def getShapModel(self, model):
        # Define function to wrap model to transform data to tensor
        shap_model = lambda x: model.cpu()(Variable( torch.from_numpy(x).float())).detach().numpy()  # Convert to float here
        return shap_model

    def calcShap(self, shap_data, explain_data, shap_model):
        # Wrap your PyTorch model with SHAP explainer
        explainer = shap.Explainer(shap_model, explain_data)
        # Compute SHAP values
        return explainer.shap_values(shap_data)

    def _save_object(self, obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    def _load_object(self, filename):
        with open(filename, 'rb') as inp:
            return pickle.load(inp)
        
    def saveShap(self, shap_values):
        # Reduce dimensionality
        shap_values_red = np.mean(np.abs(shap_values), axis=2)
        # Save the SHAP values
        self._save_object(shap_values_red, f'shap_values{FILE_DELIN}.pkl')
        return shap_values_red
        
    def outputDataframe(self, adata, shap_values_red):
        importance_df = pd.DataFrame({
            "gene": adata.var_names,
            "mean_abs_shap": np.mean(np.abs(shap_values_red), axis=0), 
            "stdev_abs_shap": np.std(np.abs(shap_values_red), axis=0)
        })
        # Save the files
        importance_df = importance_df.set_index("gene")
        importance_df['shap_percent'] = importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].sum() * 100
        importance_df = importance_df.sort_values("shap_percent", ascending=False)
        importance_df.to_csv(f'shap_values{self.FILE_DELIN}.csv', index=True) # summarised data
        
    def plotShapValues(self, shap_values, shap_data, adata):
        # Show and save the summary feature importance plot
        plt.figure()
        shap.summary_plot(np.mean(shap_values, axis=2)*-1, features=shap_data, feature_names=adata.var_names, show=False)
        plt.savefig(f'Feature_importance_summary{self.FILE_DELIN}.png', dpi=600)
        plt.show()

    def runShap(self):
        shap_data = self.getShapData(self.X_test)
        explain_data, shap_data = self.getShapSelection(shap_data)
        shap_model = self.getShapModel(self.model)
        shap_values = self.calcShap(shap_data, explain_data, shap_model)
        shap_values_red = self.saveShap(shap_values)
        self.outputDataframe(self.adata, shap_values_red)
        self.plotShapValues(shap_values, shap_data, self.adata)
