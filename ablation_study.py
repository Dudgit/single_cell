import gc
import os
from omegaconf import OmegaConf
import torch
from models.REGINA.train import phase1, phase2
from models.REGINA.dataloader import get_loaders
from models.REGINA.model import REGINA

import numpy as np
dataset_name = "norman"

def get_Regina(study_id):
    checkpointRoot = os.path.join('saved_models',dataset_name,"REGINA_"+study_id)
    cpkt_path = os.path.join(checkpointRoot, "best-all_gene_pearson.ckpt")
    model = REGINA.load_from_checkpoint(cpkt_path, weights_only=False)
    return model

def converter(gene_name):
    if "ctrl" not in gene_name:
        return gene_name
    if "ctrl+" in gene_name:
        return gene_name.split("+")[1]
    if "+ctrl" in gene_name:
        return gene_name.split("+")[0]


def eval_combo_with_REGINA(model, condition_name, ctrl_adata, num_samples=100, seq_len=16, gene_to_idx=None, device='cpu', idx = 0):
    """
    Evaluates REGINA on single or multi-gene perturbations.
    condition_name: Can be a single gene ("KLF1"), a combo string ("KLF1+CEBPA"), or a list (['KLF1', 'CEBPA']).
    """
    # 1. Parse the Combinatorial Condition
    print(f"Evaluating condition: {condition_name}, Index: {idx}/107")
    if isinstance(condition_name, str):
        genes = condition_name.split('+') # Handles Norman dataset formatting
    else:
        genes = list(condition_name)
        
    # 2. Map all targets to their vocabulary indices
    try:
        target_indices = [gene_to_idx[g] for g in genes]
    except KeyError as e:
        raise ValueError(f"Gene {e} not found in the gene_to_idx vocabulary!")

    # 3. Sample and Pad Control Cells
    sampled_idx = np.random.choice(ctrl_adata.shape[0], num_samples, replace=False)
    x_ctrl = ctrl_adata.X[sampled_idx]
    
    # Handle sparse matrices safely
    if hasattr(x_ctrl, 'toarray'):
        x_ctrl = x_ctrl.toarray()
        
    num_paddings = (seq_len - (ctrl_adata.shape[1] % seq_len)) % seq_len
    if num_paddings > 0:
        x_ctrl = np.concatenate([x_ctrl, np.zeros((num_samples, num_paddings), dtype=np.float32)], axis=1)
    
    x_ctrl = x_ctrl.reshape(num_samples, seq_len, -1)
    
    # 4. Create the Multi-Gene Index Tensor
    # Shape becomes (num_samples, num_genes_in_combo) instead of a 1D list
    gene_idx_batch = [target_indices for _ in range(num_samples)]
    gene_idx_tensor = torch.tensor(gene_idx_batch, dtype=torch.long, device=device)
    x_ctrl_tensor = torch.from_numpy(x_ctrl).float().to(device)
    
    # 5. Inference Pass (Wrapped in no_grad for memory efficiency)
    model.eval()
    with torch.no_grad():
        z_ctrl = model.encoder(x_ctrl_tensor)
        
        # Pass the 2D tensor of target indices to the prompt function
        z_prompt_fwd = model.get_perturbation_prompt(x_ctrl_tensor, gene_idx_tensor)
        
        delta_fwd = model.transition_fwd(z_ctrl, z_prompt_fwd)
        z_fake_pert = z_ctrl + delta_fwd
        
        x_recon = model.decoder(z_fake_pert).cpu().numpy()
        
    # 6. Unpad and Return
    # Using python slicing logic to safely remove padding
    X_true_recon = x_recon.reshape(num_samples, -1)
    if num_paddings > 0:
        X_true_recon = X_true_recon[:, :-num_paddings]
    return X_true_recon


def train_once(sub_cfg, study_id = None):
    run_name = dataset_name
    checkpointRoot = os.path.join('saved_models',dataset_name,"REGINA_"+study_id)
    os.makedirs(checkpointRoot, exist_ok=True)
    trainloader, valloader,*_ = get_loaders(run_name,batch_size=sub_cfg.batch_size,perturb_key=sub_cfg.perturb_key,ctrl_key=sub_cfg.ctrl_key,seq_length=16)
    phase1(sub_cfg,trainloader,valloader,checkpointRoot,run_name=run_name)
    gc.collect()
    torch.cuda.empty_cache()
    phase2(sub_cfg,trainloader,valloader,checkpointRoot,run_name=run_name)

from models.REGINA.inference import get_genes
import pickle
import anndata as ad
def infer_REGINAS(study_id):
    global_rank = int(os.environ.get("RANK", 0))
    if global_rank != 0:
        return # Ranks 1, 2, and 3 will silently exit the function here
    num_samples = 100
    seq_len = 16
    with open(f"data/{dataset_name}/gene_to_idx.pkl", "rb") as f:
        gene_to_idx = pickle.load(f)
    ctrl_adata = ad.read_h5ad(f"data/{dataset_name}/ctrl.h5ad")
    genes = get_genes(dataset_name, perturb_key="condition")

    model = get_Regina(study_id)
    model = model.eval()
    model = model.to('cpu')
    genes_inp = [converter(gene) for gene in genes]
    results_X = [eval_combo_with_REGINA(model, gene, ctrl_adata, gene_to_idx=gene_to_idx, num_samples=num_samples, seq_len=seq_len,idx=i) for i,gene in enumerate(genes_inp)]
    resultsX = np.array(results_X)
    resultsX = resultsX.reshape(-1,resultsX.shape[-1])
    resAnndata = ad.AnnData(X = resultsX)
    resAnndata.obs["gene"] = np.array([[gene] * num_samples for gene in genes_inp]).flatten()
    resAnndata.write_h5ad(os.path.join('outputs',f'{dataset_name}',f"REGINA_{study_id}.h5ad"))


def main(sub_cfg):
    zero_out_parameters = [
        "lossClassFactr" ,
        "adversarialFactr",
        "centerFactr"
                           ]
    study_ids = {"centerFactr":"no_center", "lossClassFactr":"no_class_loss", "adversarialFactr":"no_adversarial"}
    og_config = sub_cfg.copy()
    #study_id = "simple_prompt"
    for param in zero_out_parameters:
        sub_cfg = og_config.copy()
        print(sub_cfg['model_kwargs']["simple_prompt"])
        sub_cfg['model_kwargs'][param] = 0.0
        print('After setting:', sub_cfg['model_kwargs'][param])
        study_id = study_ids[param]
        train_once(sub_cfg, study_id=study_id)
    infer_REGINAS(study_id)
    for param in zero_out_parameters:
        study_id = study_ids[param]
        infer_REGINAS(study_id)
    


if __name__ == "__main__":
    cfg = OmegaConf.load('config.yaml')
    sub_cfg = cfg.get('Regina_kwargs').get("norman")
    main(sub_cfg)
