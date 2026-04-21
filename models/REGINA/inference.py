from models.REGINA.model import REGINA
import os
import pickle
import numpy as np
import anndata as ad
import torch

data_dir = "data"

def get_genes(dataset_name,perturb_key):
    dataPath = os.path.join(data_dir, dataset_name,"test.h5ad")
    test_adata = ad.read_h5ad(dataPath)
    genes = test_adata.obs[perturb_key].unique().tolist()
    return genes


def get_Regina(dataset_name):
    checkpointRoot = os.path.join('saved_models', dataset_name, "REGINA")
    cpkt_path = os.path.join(checkpointRoot, "best-all_gene_pearson.ckpt")
    model = REGINA.load_from_checkpoint(cpkt_path, weights_only=False)
    return model

def eval_gene_with_REGINA(model,gene_name,ctrl_adata,num_samples=100, seq_len=16,gene_to_idx=None):
    
    x_ctrl = ctrl_adata.X[np.random.choice(ctrl_adata.shape[0], num_samples, replace=False)].toarray()
    num_paddings = (seq_len - (ctrl_adata.shape[1] % seq_len)) % seq_len
    x_ctrl = np.concatenate([x_ctrl, np.zeros((num_samples, num_paddings), dtype=np.float32)], axis=1)
    x_ctrl = x_ctrl.reshape(num_samples, seq_len, -1)
    
    gene_idx = [gene_to_idx[gene_name]]*num_samples 
    
    z_ctrl = model.encoder(torch.from_numpy(x_ctrl).float())
    z_prompt_fwd = model.get_perturbation_prompt(torch.from_numpy(x_ctrl).float(), torch.tensor(gene_idx))
    delta_fwd = model.transition_fwd(z_ctrl, z_prompt_fwd)
    z_fake_pert = z_ctrl + delta_fwd
    x_recon = model.decoder(z_fake_pert).detach().numpy()
    X_true_recon = x_recon.reshape(num_samples, -1)[:,:-num_paddings]
    
    return X_true_recon


def inference(dataset_name,num_samples=100, seq_len=16):
    
    with open(f"data/{dataset_name}/gene_to_idx.pkl", "rb") as f:
        gene_to_idx = pickle.load(f)
    
    ctrl_adata = ad.read_h5ad(f"data/{dataset_name}/ctrl.h5ad")
    genes = get_genes(dataset_name, perturb_key="condition")
    genes_inp = [gene.split("+")[0] if "+" in gene else gene for gene in genes]   
    
    model = get_Regina(dataset_name)
    model = model.eval()
    model = model.to('cpu')
    
    results_X = [eval_gene_with_REGINA(model, gene_name, ctrl_adata,gene_to_idx=gene_to_idx, num_samples=num_samples, seq_len=seq_len) for gene_name in genes_inp]
    resultsX = np.array(results_X)
    resultsX = resultsX.reshape(-1,resultsX.shape[-1])
    resAnndata = ad.AnnData(X = resultsX)
    resAnndata.obs["gene"] = np.array([[gene] * num_samples for gene in genes_inp]).flatten()
    resAnndata.write_h5ad(os.path.join('outputs',f'{dataset_name}',"REGINA.h5ad"))