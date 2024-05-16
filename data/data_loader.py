import scanpy as sc
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self, adata, device, test_split, batch_size, num_workers) -> None:
        self.adata = adata
        self.device = device
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.shap_data = None
    
    def adata_to_tensor(self, adata, device):
        x = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(device)
        y = torch.tensor(adata.obs['condition'], dtype=torch.int8).to(device)
        return x, y

    def split_data(self, x, y, test_split):
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, train_size=test_split)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.5)
        del x_temp, y_temp
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def create_dataset(self, x_train, x_val, x_test, y_train, y_val, y_test):
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        val_dataset = TensorDataset(x_val, y_val)
        return train_dataset, test_dataset, val_dataset
    
    def create_dataloader(self, train_dataset, test_dataset, val_dataset, batch_size, num_workers):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))
        return train_dataloader, test_dataloader, val_dataloader
    
    def write_shap_data(self, x_test):
        return x_test.cpu().numpy().astype(np.float32).copy()
    
    def get_shap_data(self):
        if self.shap_data is None:
            raise ValueError('No shap data found. Please run get_dataloaders() first.')
        return self.shap_data
    
    def get_dataloaders(self):
        # Convert adata to two tensors, based on .X array and .obs[condition] column
        x, y = self.adata_to_tensor(self.adata, self.device)
        # Split the data based on test split, val and test are 50/50
        x_train, x_val, x_test, y_train, y_val, y_test = self.split_data(
            x, y, self.test_split)
        # Load the data into PyTorch TensorDataset
        train_dataset, test_dataset, val_dataset = self.create_dataset(
            x_train, x_val, x_test, y_train, y_val, y_test)
        # Load the data into PyTorch DataLoader
        train_dataloader, test_dataloader, val_dataloader = self.create_dataloader(
            train_dataset, test_dataset, val_dataset, self.batch_size, self.num_workers)
        # Save x_train as shap_data in the cpu for later evaluation
        self.shap_data = self.write_shap_data(x_test)
        # Free gpu memory
        del x_train, y_train, x_val, x_test, y_val, y_test
        del train_dataset, test_dataset, val_dataset
        # Return the dataloaders for training
        return train_dataloader, test_dataloader, val_dataloader
