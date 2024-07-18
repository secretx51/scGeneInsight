import scanpy as sc
import pandas as pd
from itertools import chain
from sklearn.preprocessing import LabelEncoder

class PreProcess_Anndata():
    def __init__(self,
                cell_type,
                key_genes, 
                gene_set,
                quantiles, 
                h5_path: str, 
                gsea_path: str, 
                file_delin: str,
                filter_key: bool,
                rem_non_coding: bool,
                memory_limit: bool) -> None:

        self.cell_type = cell_type
        self.key_genes = key_genes
        self.gene_set = gene_set
        self.quantiles = quantiles
        
        self.h5_path = h5_path
        self.gsea_path = gsea_path
        self.file_delin = file_delin
        self.filter_key = filter_key
        self.rem_non_coding = rem_non_coding
        self.memory_limit = memory_limit
        
        self.le_dict = None
    
    def read_adata(self, adata_path):
        return sc.read_h5ad(adata_path)
    
    def filter_cells(self, adata, cell_type):
        return adata[adata.obs['super_cell_type'] == cell_type]
    
    def remove_non_coding(self, adata):
        return adata[:, ~(adata.var_names.str.startswith('A') & adata.var_names.str.contains('.'))]
    
    def read_gsea(self, path):
        return pd.read_csv(path, index_col=0)
    
    def filter_gsea(self, df, term):
        return df[df['Term'] == term]
    
    def adata_term(self, adata, df):
        adata.obs = adata.obs.join(df[['ES']])
        return adata
    
    def quantile_dict(self, adata, quantiles):
        return {f'p{int(quantile*100)}': adata.obs['ES'].quantile(quantile) for quantile in quantiles}
    
    def adata_condition(self, adata, quantile_dict):
        def assign_condition(es):
            for percentile, cutoff in quantile_dict.items():
                if es <= cutoff:
                    return percentile
            return 'p100'
        # Apply the above function to the ES column
        adata.obs['condition'] = adata.obs['ES'].apply(assign_condition)
        return adata
        
    def encode_condition(self, adata):
        le = LabelEncoder()
        adata.obs['condition'] = le.fit_transform(adata.obs['condition'])
        return adata, le
    
    def calc_le_dict(self, le):
        le_classes = le.classes_
        le_labels = le.transform(le_classes)
        return dict(zip(le_classes, le_labels))
    
    # Define a function to read and parse the text file
    def read_gmt(self, path):
        data_dict = {}  # Initialize an empty dictionary to store the data
        with open(path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                # Check if there are at least two items (key and one value)
                if len(parts) >= 2:
                    key = str(parts[0])  # The first item is the key
                    values = parts[2:]  # The rest are values except description
                    # Store the key and values in the dictionary
                    data_dict[key] = values
        return data_dict
    
    def filter_genes(self, adata, gmt):
        gmt_genes = set(chain.from_iterable(gmt.values()))
        selected_genes = adata.var.highly_variable | adata.var.index.isin(gmt_genes)
        return adata[:, selected_genes]
    
    def filter_adata(self, adata_path, cell_type, rem_non_coding):
        adata = self.read_adata(adata_path)
        adata = self.filter_cells(adata, cell_type)
        if rem_non_coding:
            adata = self.remove_non_coding(adata)
        return adata
    
    def add_gsea(self, adata, gsea_path, term):
        gsea_df = self.read_gsea(gsea_path)
        gsea_df = self.filter_gsea(gsea_df, term)
        adata = self.adata_term(adata, gsea_df)
        return adata
    
    def encode_adata(self, adata, quantiles):
        quantile_dict = self.quantile_dict(adata, quantiles)
        adata = self.adata_condition(adata, quantile_dict)
        adata, le = self.encode_condition(adata)
        self.le_dict = self.calc_le_dict(le)
        return adata
    
    def get_le_dict(self):
        if self.le_dict is None:
            raise ValueError('le_dict is not defined, use preprocess_adata first')
        return self.le_dict
    
    def reduce_adata(self, adata, path, memory_limit):
        if not memory_limit:
            return adata
        gmt = self.read_gmt(path)
        adata = self.filter_genes(adata, gmt)
        return adata
    
    def filter_key_genes(self, adata, key_genes, filter_key):
        return adata[:, ~adata.var_names.isin(key_genes)] if filter_key else adata
        
    def preprocess_adata(self):
        adata = self.filter_adata(self.h5_path, self.cell_type, self.rem_non_coding)
        adata = self.add_gsea(adata, self.gsea_path, self.gene_set)
        adata = self.encode_adata(adata, self.quantiles)
        adata = self.reduce_adata(adata, self.gsea_path, self.memory_limit)
        adata = self.filter_key_genes(adata, self.key_genes, self.filter_key)
        print(adata)
        return adata
