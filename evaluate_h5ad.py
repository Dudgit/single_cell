from eval_metrics import PrecisionEvaluator, DistributionEvaluator

from omegaconf import OmegaConf
import anndata as ad
import pandas as pd
import numpy as np
import torch
import os
import pickle
import scipy.sparse as sp
cfg = OmegaConf.load("config.yaml")
from models.GEARS.inference import inference_gears
from models.REGINA.inference import inference as inference_REGINA

def get_gt_data(dataset_name,perturb_key = "perturbation",ctrl_key = "ctrl"):
    gt_path = os.path.join('data', dataset_name, 'test.h5ad')
    gt = ad.read_h5ad(gt_path)
    ctrl_path = os.path.join('data', dataset_name, 'train.h5ad')
    ctrl = ad.read_h5ad(ctrl_path)
    ctrl_true = ctrl[ctrl.obs[perturb_key] == ctrl_key]
    return gt, ctrl_true

def eval_gene(gene,gt,ctrl_true,res,pe,de,num_samples,perturb_key):
    ctrl = ctrl_true.X[np.random.choice(ctrl_true.shape[0], num_samples, replace=False)]
    
    if sp.issparse(ctrl):
        ctrl = ctrl.toarray()
    
    pert = gt[gt.obs[perturb_key] == gene].X.toarray()
    
    if sp.issparse(pert):
        pert = pert.toarray()

    if pert.shape[0] > num_samples:
        pert = pert[np.random.choice(pert.shape[0], num_samples, replace=False)]
    
    pred = res[res.obs['gene'] == gene.split('+')[0]].X    
    precision_scores = pe.evaluate_batch(torch.tensor(pert), torch.tensor(ctrl), torch.tensor(pred))
    distribution_scores = de.evaluate_batch(torch.tensor(pert), torch.tensor(pred))
    return precision_scores, distribution_scores


def evaluate_h5ad(res_path,dataset_name,model_name,perturb_key = "perturbation",ctrl_key = "ctrl",num_samples=100):
    #* Loading in h5ad files
    print("Loading ground truth and control data...")
    gt, ctrl_true = get_gt_data(dataset_name, perturb_key, ctrl_key)
    res = ad.read_h5ad(res_path)
    genes = gt.obs[perturb_key].unique()
    
    print("Data loaded. Starting evaluation...")
    #* Creating evaulators
    pe = PrecisionEvaluator()
    de = DistributionEvaluator()

    print("Evaluators created. Evaluating each gene...")
    #* Results metric dictionaries
    precision_results ={"pearson_top5": [], "pearson_all": [], "direction_error_top5": [], "direction_error_all": [], "nmse_top5": [], "nmse_all": []}
    distribution_results = {"MMD": [], "Wasserstein": [],"Variance_preservation": [],"Energy_loss": [], "L1_distance": []}
    
    print("Evaluating genes...")
    for gene in genes:
        pscore, dscore = eval_gene(gene,gt=gt, ctrl_true=ctrl_true, res=res, pe=pe, de=de, num_samples=num_samples, perturb_key=perturb_key)
        for i, key in enumerate(precision_results.keys()):
            precision_results[key].append(pscore[i])
        for i, key in enumerate(distribution_results.keys()):
            distribution_results[key].append(dscore[i])
    pickle.dump(precision_results, open(f'outputs/{dataset_name}/{model_name}_precision.pkl', 'wb'))
    pickle.dump(distribution_results, open(f'outputs/{dataset_name}/{model_name}_distribution.pkl', 'wb'))

if __name__ == "__main__":
    dataset_name = "dixit"
    model_name = "REGINA"
    num_samples = 100
    
    cfg = OmegaConf.load('config.yaml')
    
    perturb_key = cfg.pert_keys.get(dataset_name)
    ctrl_key = cfg.ctrl_keys.get(dataset_name)

    print(f"Starting inference for {model_name}...")

    inference_REGINA(dataset_name,num_samples=num_samples, seq_len=16)
    
    print("Inference completed. Starting evaluation...")
    
    res_path = os.path.join('outputs',f'{dataset_name}',"REGINA.h5ad")
    
    evaluate_h5ad(res_path=res_path,dataset_name=dataset_name,model_name=model_name,perturb_key=perturb_key,ctrl_key=ctrl_key,num_samples=num_samples)