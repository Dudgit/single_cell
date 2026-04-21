import numpy as np
import anndata as ad
import torch
import pandas as pd
import scipy.sparse
import pickle

class CellDataset(torch.utils.data.Dataset):
    def __init__(self, adataTarget, adataInput, gene_to_idx, seq_length: int = 32,num_paddings: int = 19):
        if scipy.sparse.isspmatrix_csc(adataTarget.X):
            self.target_X = adataTarget.X.tocsr()
        else:
            self.target_X = adataTarget.X

        if scipy.sparse.isspmatrix_csc(adataInput.X):
            self.input_X = adataInput.X.tocsr()
        else:
            self.input_X = adataInput.X

        self.target_genes = adataTarget.obs['gene'].values  # Array of strings
        self.target_states = adataTarget.obs.iloc[:,-3:].to_numpy(dtype=np.float32)
        self.input_states = adataInput.obs.iloc[:,-3:].to_numpy(dtype=np.float32)
        
        self.gene_to_idx = gene_to_idx
        self.seq_length = seq_length
        self.num_inputs = self.input_X.shape[0]
        self.num_genes = len(gene_to_idx)
        self.num_paddings = num_paddings
        
    def __len__(self):
        return self.target_X.shape[0]

    def __getitem__(self, idx):
        # 1. Fast Sparse Slicing (CSR is O(1) for rows)
        data_point = self.target_X[idx]

        # Check if it has the toarray method (is sparse)
        if hasattr(data_point, 'toarray'):
            data_point = data_point.toarray()

        x_row = data_point.ravel()
        x = np.concatenate([x_row, np.zeros(self.num_paddings, dtype=np.float32)]) 
        
        # 2. Random input selection
        inputidx = np.random.randint(0, self.num_inputs)
        x_in_row = self.input_X[inputidx].toarray().ravel()
        x_input = np.concatenate([x_in_row, np.zeros(self.num_paddings, dtype=np.float32)])
        
        # 3. Fast Numpy Lookup (No Pandas overhead)
        gene_name = self.target_genes[idx]
        gene_idx = self.gene_to_idx[gene_name]
        
        state = self.target_states[idx]
        input_state = self.input_states[inputidx]

        x = x.astype(np.float32)
        x_input = x_input.astype(np.float32)
        
        return (x.reshape(self.seq_length, -1), 
                x_input.reshape(self.seq_length, -1), 
                gene_idx, 
                state, 
                input_state)


class CellDataset_binary(torch.utils.data.Dataset):
    def __init__(self, adataTarget, adataInput, gene_to_idx, seq_length: int = 32,perturb_key: str = "condition"):
        if scipy.sparse.isspmatrix_csc(adataTarget.X):
            self.target_X = adataTarget.X.tocsr()
        else:
            self.target_X = adataTarget.X

        if scipy.sparse.isspmatrix_csc(adataInput.X):
            self.input_X = adataInput.X.tocsr()
        else:
            self.input_X = adataInput.X

        self.target_genes = adataTarget.obs[perturb_key].apply(lambda x: x.split("+")[0]).values  # Array of strings
        self.target_states = adataTarget.obs.iloc[:,-1].to_numpy(dtype=np.float32)
        self.input_states = adataInput.obs.iloc[:,-1].to_numpy(dtype=np.float32)
        
        self.gene_to_idx = gene_to_idx
        self.seq_length = seq_length
        self.num_inputs = self.input_X.shape[0]
        self.num_genes = len(gene_to_idx)
        ### Calculate num paddings:
        self.num_paddings = (self.seq_length - (self.num_genes % self.seq_length)) % self.seq_length
        print(f"Calculated num_paddings: {self.num_paddings} for seq_length: {self.seq_length} and num_genes: {self.num_genes}")
        #self.num_paddings = num_paddings
        
    def __len__(self):
        return self.target_X.shape[0]

    def __getitem__(self, idx):
        # 1. Fast Sparse Slicing (CSR is O(1) for rows)
        data_point = self.target_X[idx]

        # Check if it has the toarray method (is sparse)
        if hasattr(data_point, 'toarray'):
            data_point = data_point.toarray()

        x_row = data_point.ravel()
        x = np.concatenate([x_row, np.zeros(self.num_paddings, dtype=np.float32)]) 
        
        # 2. Random input selection
        inputidx = np.random.randint(0, self.num_inputs)
        x_in_row = self.input_X[inputidx].toarray().ravel()
        x_input = np.concatenate([x_in_row, np.zeros(self.num_paddings, dtype=np.float32)])
        
        # 3. Fast Numpy Lookup (No Pandas overhead)
        gene_name = self.target_genes[idx]
        gene_idx = self.gene_to_idx[gene_name]
        
        state = self.target_states[idx]
        input_state = self.input_states[inputidx]

        x = x.astype(np.float32)
        x_input = x_input.astype(np.float32)
        
        return (x.reshape(self.seq_length, -1), 
                x_input.reshape(self.seq_length, -1), 
                gene_idx, 
                state, 
                input_state)


def get_loaders(dataset_name,batch_size:int,num_workers:int = 8,ctrl_key:str = "ctrl",perturb_key:str = "condition", seq_length: int = 32):
    train_data = ad.read_h5ad(f"data/{dataset_name}/train_processed.h5ad")
    val_data = ad.read_h5ad(f"data/{dataset_name}/val_processed.h5ad")
    test_data = ad.read_h5ad(f"data/{dataset_name}/test_processed.h5ad")
    ctrl = train_data[train_data.obs[perturb_key] == ctrl_key]
    with open(f"data/{dataset_name}/gene_to_idx.pkl", "rb") as f:
        gene_to_idx = pickle.load(f)
    idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
    pin_memory = True if num_workers > 0 else False
    persistent_workers = True if num_workers > 0 else False

    train_set = CellDataset_binary(train_data[train_data.obs[perturb_key] != ctrl_key], ctrl, gene_to_idx,seq_length=seq_length, perturb_key=perturb_key)
    val_set = CellDataset_binary(val_data[val_data.obs[perturb_key] != ctrl_key], ctrl, gene_to_idx,seq_length=seq_length, perturb_key=perturb_key)
    test_set = CellDataset_binary(test_data[test_data.obs[perturb_key] != ctrl_key], ctrl, gene_to_idx,seq_length=seq_length, perturb_key=perturb_key)

    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}, Test set size: {len(test_set)}")
    trainloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size,shuffle= True,num_workers=num_workers,persistent_workers=persistent_workers,pin_memory=pin_memory,drop_last=False)
    valloader = torch.utils.data.DataLoader(val_set, batch_size = batch_size ,shuffle= False,num_workers=num_workers,persistent_workers=persistent_workers,pin_memory=pin_memory,drop_last=False    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size,shuffle= False,num_workers=num_workers,persistent_workers=persistent_workers,pin_memory=pin_memory)
    
    return trainloader, valloader, test_loader, gene_to_idx, idx_to_gene
