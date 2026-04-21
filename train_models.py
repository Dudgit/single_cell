
#!
#! This is just the pseudo-code
#!

import gc
import os
from omegaconf import OmegaConf
from models.GEARS.train import train_gears
import pandas as pd
pd.Series.nonzero = lambda self: self.to_numpy().nonzero()

from models.REGINA.train import phase1, phase2
from models.REGINA.dataloader import get_loaders
import torch
#from models.PERTURBNET import train as train_PERTURBNET

def train(model_name,dataset_name,cfg):
    
    if model_name == "GEARS":
        hidden_size = cfg.get('gears_kwargs').get(dataset_name).hidden_size
        train_gears(dataset_name=dataset_name,hidden_size=hidden_size)
    
    elif model_name == "REGINA":
        run_name = dataset_name
        checkpointRoot = os.path.join('saved_models',run_name,"REGINA")
        os.makedirs(checkpointRoot, exist_ok=True)
        sub_cfg = cfg.get('Regina_kwargs').get(run_name)
        trainloader, valloader,*_ = get_loaders(run_name,batch_size=sub_cfg.batch_size,perturb_key=sub_cfg.perturb_key,ctrl_key=sub_cfg.ctrl_key,seq_length=16)
        phase1(sub_cfg,trainloader,valloader,checkpointRoot,run_name=run_name)
        gc.collect()
        torch.cuda.empty_cache()
        phase2(sub_cfg,trainloader,valloader,checkpointRoot,run_name=run_name,phase_2_epochs=10)


if __name__ == "__main__":
    cfg = OmegaConf.load('config.yaml')
    run_name = "norman"
    train(model_name="GEARS",dataset_name=run_name,cfg=cfg)
    