import os
from pytorch_lightning.plugins.environments import LightningEnvironment
import gc
from models.REGINA.model import REGINA
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.loggers import WandbLogger
import torch
torch.set_float32_matmul_precision('medium')
import numpy as np    
wandbOnline = True
from omegaconf.listconfig import ListConfig

def phase1(cfg,trainloader,valloader,model_directory_path,run_name="adamson"):
    print('Starting Phase 1 Training')
    modelKwargs = cfg.model_kwargs
    cfg.run_name = run_name
    model = REGINA(phase=1,**modelKwargs)    
    max_var = ModelCheckpoint(monitor="val/Health_Var_Preservation",mode="max",save_top_k=1,filename="best-var_pres",dirpath=model_directory_path)
    print("Model and Logger Initialized, Starting Training")
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,accelerator="auto",devices="auto" ,callbacks=[max_var]#,logger=logger
                        ,strategy="ddp_find_unused_parameters_true",plugins=LightningEnvironment())
    trainer.fit(model,train_dataloaders=trainloader, val_dataloaders=valloader)
    del trainer
    del model
    # Wait for all GPUs to finish before destroying
    torch.distributed.barrier() if torch.distributed.is_initialized() else None

def phase2(cfg,trainloader,valloader,model_directory_path,run_name,phase_2_epochs = None):
    cfg.run_name = run_name
    print('Starting Phase 2 Training')
    best_all_pearson = ModelCheckpoint(monitor="val/Pearson_All",mode="max",save_top_k=1,filename="best-all_gene_pearson",dirpath=model_directory_path)
    torch.serialization.add_safe_globals([ListConfig])
    
    cpkt_path = os.path.join(model_directory_path, "best-var_pres.ckpt")
    model = REGINA.load_from_checkpoint(cpkt_path, weights_only=False)
    model.init_phase2()
    max_epochs = phase_2_epochs if phase_2_epochs is not None else cfg.max_epochs
    trainer = pl.Trainer(max_epochs=max_epochs,strategy="ddp_find_unused_parameters_true",num_sanity_val_steps=0,callbacks=[best_all_pearson]#,logger=logger
                         ,plugins=LightningEnvironment())
    trainer.fit(model,train_dataloaders=trainloader, val_dataloaders=valloader)