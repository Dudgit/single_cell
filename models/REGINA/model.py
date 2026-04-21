import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np  
import torchmetrics
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MulticlassConfusionMatrix
import torchmetrics
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.functional import pearson_corrcoef
import torch.nn.functional as F
from models.REGINA.utils import CenterLoss

from models.REGINA.modules import TransformerCycleEncoder, TransformerCycleDecoder,  LatentClassifier, SemanticDiscriminator
from models.REGINA.modules import DeltaTransitionv2 as DeltaTransition

class REGINA(pl.LightningModule):
    def __init__(self, input_dim=317, d_model=256, n_layers=4, n_heads=8, z_dim=256
                ,reconFactr = 1.5, lossLatentFactr = 1.0, lossCycleFactr = 5., lossRegFactr = 1e-4
                , lossClassFactr=1.0, centerFactr=0.5, useVarLoss:bool = False,semanticCycleFactr = 2.0,
                adversarialFactr = 3.5, mmdFactr = 100.0,directionalFactr = 0.1,pearsonfactr = 1.0,
                classes = ['class1', 'class2', 'class3', 'class4'],
                phase:int = 2, n_real_genes:int=5060, seq_len:int=64,phase2reconfactor= 1.,
                cond_dim = 256, silence_val = -1,
                use_hvg_mask:bool = False # This is new
                ):
        super().__init__()
        
        self.phase = phase
        #Actual Training and hyperparameters        
        ##Loss Factors
        total_input_dim = input_dim*seq_len
        self.reconFactr = reconFactr
        self.lossLatentFactr = lossLatentFactr
        self.lossCycleFactr = lossCycleFactr
        self.lossRegFactr = lossRegFactr
        self.LossClassFactr = lossClassFactr
        self.centerFactr = centerFactr
        self.semanticCycleFactr = semanticCycleFactr
        self.adversarialFactr = adversarialFactr
        self.mmdFactr = mmdFactr
        self.directionalFactr = directionalFactr
        self.pearsonfactr = pearsonfactr
        self.phase2reconfactor = phase2reconfactor

        #Model parameters
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.useVarLoss = useVarLoss
        num_classes = len(classes)
        self.num_classes = num_classes
        #Model Components
        self.encoder = TransformerCycleEncoder(input_dim, d_model, n_layers, n_heads, z_dim, seq_len=seq_len)
        self.decoder = TransformerCycleDecoder(input_dim, d_model, n_layers, n_heads, z_dim, seq_len=seq_len)
        self.transition_fwd = DeltaTransition(z_dim)
        self.transition_bwd = DeltaTransition(z_dim)
        self.latentClassifier = LatentClassifier(z_dim, num_classes)

        self.discriminator = SemanticDiscriminator(z_dim)
        self.silence_val = silence_val  # Learnable silence value for perturbation prompt
        mask = torch.zeros(total_input_dim)
        mask[:n_real_genes] = 1.0
        mask = mask.view(1, seq_len, input_dim) 
        self.register_buffer('gene_mask', mask)
        if use_hvg_mask:
            hvg_mask = np.load("data/genes_to_predict.txt", dtype=str)
            padding = total_input_dim - len(hvg_mask)
            hvg_mask = np.concatenate([np.isin(self.gene_mask.squeeze().cpu().numpy(), hvg_mask), np.zeros(padding)])
            hvg_mask_tensor = torch.tensor(hvg_mask, dtype=torch.float32).view(1, seq_len, input_dim)
            self.gene_mask = self.gene_mask * hvg_mask_tensor
        #Traning phase settings
        if phase == 1:
            self.automatic_optimization = False
            self.shared_step = self.phase_one_shared_step
            for param in self.transition_fwd.parameters():
                param.requires_grad = False
            for param in self.transition_bwd.parameters():
                param.requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = False
            self.transition_fwd.eval()
            self.transition_bwd.eval()
            self.discriminator.eval()


        
        #Losses
        self.criterion = nn.MSELoss()
        self.latent_criterion = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss(
            num_classes=num_classes,
            feat_dim=z_dim,
            use_gpu=torch.cuda.is_available(),
        )
        # Checking metrics and classifiers
    
        self.aucMetric = torchmetrics.AUROC(num_classes=num_classes, average=None,task="multiclass")
        self.confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        self.confmat2 = MulticlassConfusionMatrix(num_classes=num_classes)
        self.class_names = classes
        self.classwise_auc = ClasswiseWrapper(
            torchmetrics.AUROC(num_labels=num_classes, average=None, task="multilabel"),
            labels=list(classes),
            prefix="Classification/val/AUROC_" # This automatically formats your output names!
        )
        self.save_hyperparameters()

    def init_phase2(self,lossCycleFactr=None, semanticCycleFactr=None, adversarialFactr=None, mmdFactr=None):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.latentClassifier.parameters():
            param.requires_grad = False
        for param in self.center_loss.parameters():
            param.requires_grad = False
        for param in self.transition_fwd.parameters():
            param.requires_grad = True
        for param in self.transition_bwd.parameters():
            param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = True
        
        if lossCycleFactr is not None:
            self.lossCycleFactr = lossCycleFactr
        if semanticCycleFactr is not None:
            self.semanticCycleFactr = semanticCycleFactr
        if adversarialFactr is not None:
            self.adversarialFactr = adversarialFactr
        if mmdFactr is not None:
            self.mmdFactr = mmdFactr
        self.shared_step = self.shared_step_cycle_gan
        self.configure_optimizers = self.configure_optimizers_cycle_gan
        self.encoder.eval() 
        self.latentClassifier.eval()


        self.transition_fwd.train()
        self.transition_bwd.train()
        self.discriminator.train()
        self.decoder.train()
        self.automatic_optimization = False
        self.phase = 2


    def configure_optimizers_cycle_gan(self):
        gen_params = [
                {  'params': list(self.transition_fwd.parameters()) + list(self.transition_bwd.parameters())},
                {'params': self.decoder.parameters(), 'lr': 5e-5}]
        opt_g = torch.optim.AdamW(gen_params, lr=4e-4, betas=(0.5, 0.999), weight_decay=1e-5)
        
        # Discriminator
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=4e-4, betas=(0.5, 0.999), weight_decay=1e-5)
        
        # Cosine annealing schedulers - prevents oscillation after early convergence
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.trainer.max_epochs, eta_min=1e-6)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=self.trainer.max_epochs, eta_min=1e-6)
        
        return [opt_g, opt_d], [sched_g, sched_d]

    
    def add_instance_noise(self, data, std=0.1):
        # Gradual linear decay instead of hard cutoff
        decay_start = 20
        decay_end = 80
        if self.current_epoch < decay_start:
            current_std = std
        elif self.current_epoch >= decay_end:
            return data
        else:
            progress = (self.current_epoch - decay_start) / (decay_end - decay_start)
            current_std = std * (1.0 - progress)
        noise = torch.randn_like(data) * current_std
        return data + noise
    

    def get_perturbation_prompt(self, x_ctrl, pert_multi_hot, normalized=True):
        """
        Creates a 'Prompt Vector' by silencing the perturbed genes and comparing latent states.
        Safely handles combinatorial dosages (e.g., GENE A + GENE A).
        """

        B, num_patches, patch_dim = x_ctrl.shape
        total_genes = num_patches * patch_dim  
        
        # 1. Standardize to a Multi-Hot vector covering all padded genes
        if pert_multi_hot.dim() == 1:
            full_multi_hot = torch.zeros(B, total_genes, device=self.device)
            # Use scatter_add_ instead of scatter_ to support dosage counts
            full_multi_hot.scatter_add_(1, pert_multi_hot.unsqueeze(1), torch.ones_like(pert_multi_hot.unsqueeze(1), dtype=torch.float))
        else:
            num_provided = pert_multi_hot.size(1)
            if num_provided < total_genes:
                padding = torch.zeros(B, total_genes - num_provided, device=self.device)
                full_multi_hot = torch.cat([pert_multi_hot, padding], dim=1)
            else:
                full_multi_hot = pert_multi_hot

        # 2. Reshape to match x_ctrl [B, Patches, Dim]
        multi_hot_reshaped = full_multi_hot.view(B, num_patches, patch_dim)
        
        # 3. Separate Binary Masking from Dosage Intensity
        # binary_mask ensures we never get negative keep_masks, even if dosage is 5.0
        binary_mask = (multi_hot_reshaped > 0).float()
        keep_mask = 1.0 - binary_mask
        
        # 4. Apply Dosage-Scaled Silencing
        # We MUST use a negative base value so the dosage multiplier has an effect.
        # If normalized=True (z-scored), -3.0 acts as a strong suppression signal.
        base_silence_val = self.silence_val if normalized else -5.0
        
        # Zero out the real gene, then inject the artificial dosage signal
        x_silenced = (x_ctrl * keep_mask) + (multi_hot_reshaped * base_silence_val)
        
        # 5. Generate the Prompt vector using the Encoder
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_silenced = self.encoder(x_silenced)
            
        z_prompt = z_silenced - z_ctrl
        
        return z_prompt
        
    
    def shared_step_cycle_gan(self, batch, optimizer_idx=0, mode='train'):
        x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
        
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)       # Real Control
            z_real_pert = self.encoder(x_pert)  # Real Perturbed

        # Generation step
        if optimizer_idx == 0:
            lossDict = {}
            # 1. Forward: Add perturbation
            z_prompt_fwd = self.get_perturbation_prompt(x_ctrl, pert_idx)
            z_prompt_bwd = z_prompt_fwd
            delta_fwd = self.transition_fwd(z_ctrl, z_prompt_fwd)
            z_fake_pert = z_ctrl + delta_fwd
            
            if pert_state.dim() > 1 and pert_state.size(1) > 1:
                original_labels = torch.argmax(pert_state, dim=1)
            else:
                original_labels = pert_state.long()


            delta_bwd_rec = self.transition_bwd(z_fake_pert, z_prompt_bwd)
            z_rec_ctrl = z_fake_pert + delta_bwd_rec
            
            # Loss: Cycle Consistency
            loss_cycle_ctrl = F.mse_loss(z_rec_ctrl, z_ctrl)
            lossDict['loss_cycle_ctrl'] = loss_cycle_ctrl

            # Generating fake control latent states
            delta_bwd_real = self.transition_bwd(z_real_pert, z_prompt_bwd)
            z_fake_ctrl = z_real_pert + delta_bwd_real
            
            # Check if we can reconstruct real perturbed data
            delta_fwd_rec = self.transition_fwd(z_fake_ctrl, z_prompt_fwd)
            z_rec_pert = z_fake_ctrl + delta_fwd_rec
            
            # Loss: Cycle Consistency
            loss_cycle_pert = F.mse_loss(z_rec_pert, z_real_pert)
            lossDict['loss_cycle_pert'] = loss_cycle_pert
            
            # Chek if we can fool the discriminator 
            logits_fake = self.discriminator(z_fake_pert, z_prompt_fwd)
            loss_adv = F.mse_loss(logits_fake, torch.ones_like(logits_fake))
            lossDict['loss_adv_g'] = loss_adv

            real_latent_delta = z_real_pert - z_ctrl
            loss_direction = F.huber_loss(delta_fwd, real_latent_delta, delta=1.0)
            lossDict['loss_direction'] = loss_direction

            # Decode z_fake_pert once and reuse
            x_fake_pert = self.decoder(z_fake_pert)
    
            loss_recon_anchor = (
            F.mse_loss(self.decoder(z_ctrl) * self.gene_mask, x_ctrl * self.gene_mask) +
            F.mse_loss(self.decoder(z_real_pert) * self.gene_mask, x_pert * self.gene_mask)
            )
            lossDict['loss_recon_anchor'] = loss_recon_anchor
            loss_x_recon = F.mse_loss(x_fake_pert * self.gene_mask, x_pert * self.gene_mask)
            g_loss = (self.lossCycleFactr * (loss_cycle_ctrl + loss_cycle_pert) + 
                      self.adversarialFactr * loss_adv +   # GAN loss usually has lower weight in CycleGAN
                      self.reconFactr * loss_x_recon
                      )
            lossDict["g_total_loss"] = g_loss
            self.logging_metrics(mode, lossDict)
            
            # Log AUC only during Generator step
            with torch.no_grad():
                
                
                
                self.log_collapse_metrics2(mode, z_ctrl, z_fake_pert, z_real_pert)
                # Reuse x_fake_pert from above instead of decoding z_fake_pert again
                x_recon_ctrl = self.decoder(z_fake_ctrl)
                
                if mode == "val":
                    self.log_reconstruction(mode, x_pert, x_ctrl, x_fake_pert, x_recon_ctrl)
                    self.log_pearson_metrics(mode, x_pert, x_ctrl, pert_idx)
                    #self.log_nmse_top_genes(mode, x_pert, x_ctrl, pert_idx, top_n=20)
                    #self.log_direction_error(mode, x_pert, x_ctrl, pert_idx, top_n=20)

            return g_loss

        #Discriminator
        if optimizer_idx == 1:        
            # 1. Real Data
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            logits_real = self.discriminator(self.add_instance_noise(z_real_pert), z_prompt)
            loss_real = F.mse_loss(logits_real, torch.ones_like(logits_real)*0.9)
            
            # 2. Fake Data
            with torch.no_grad():
                delta_fwd = self.transition_fwd(z_ctrl, z_prompt)
                z_fake_pert = z_ctrl + delta_fwd
            
            logits_fake = self.discriminator(self.add_instance_noise(z_fake_pert.detach()), z_prompt)
            loss_fake = F.mse_loss(logits_fake, torch.zeros_like(logits_fake))
            
            # Total Discriminator Loss
            d_loss = 0.5 * (loss_real + loss_fake)
            
            self.logging_metrics(mode, {'d_loss': d_loss, 'd_real': logits_real.mean(), 'd_fake': logits_fake.mean()})
            return d_loss

    def phase_one_shared_step(self, batch, mode):
        x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
        
        # 1. Parse Original Labels
        if pert_state.dim() > 1 and pert_state.size(1) > 1:
            original_labels = torch.argmax(pert_state, dim=1)
        else:
            original_labels = pert_state.long()
            
        loss_dict = {}
        z_ctrl = self.encoder(x_ctrl)
        z_real_pert = self.encoder(x_pert)
        x_recon_ctrl = self.decoder(z_ctrl)
        x_recon_pert = self.decoder(z_real_pert)
        
        logits = self.latentClassifier(z_real_pert)
        loss_class = self.latent_criterion(logits, original_labels)
        
        
        logits_critic = self.latentClassifier(z_real_pert.detach())
        loss_class_critic = self.latent_criterion(logits_critic, original_labels)
        
        loss_dict['loss_class_critic'] = loss_class_critic
        
        # Reconstruction & Regularization (unchanged)
        batch_size = x_ctrl.size(0)
        n_valid_elements = self.gene_mask.sum() * batch_size

        diff_ctrl = (x_recon_ctrl - x_ctrl) * self.gene_mask
        diff_pert = (x_recon_pert - x_pert) * self.gene_mask
        loss_recon_ctrl = (diff_ctrl ** 2).sum() / n_valid_elements
        loss_recon_pert = (diff_pert ** 2).sum() / n_valid_elements
        loss_recon = loss_recon_ctrl + loss_recon_pert
        loss_dict['loss_recon'] = loss_recon
        
        loss_reg = torch.mean(z_ctrl**2) + torch.mean(z_real_pert**2)
        loss_dict['loss_reg'] = loss_reg

        z_recon_pert = self.encoder(x_recon_pert)
        loss_consistency = self.criterion(z_recon_pert, z_real_pert.detach())
        loss_dict['loss_consistency'] = loss_consistency
        
        
        center_loss = self.center_loss(z_real_pert, original_labels)
        loss_dict['center_loss'] = center_loss

        total_loss = (self.reconFactr * loss_recon + 
                      self.lossRegFactr * loss_reg +
                      self.LossClassFactr * loss_class +
                      self.lossLatentFactr * loss_consistency +
                      self.centerFactr * center_loss)
        

        loss_dict['total_loss'] = total_loss
        self.logging_metrics(mode, loss_dict)
        
        if mode == "val":
            # You will need to update logAUROC to handle the two separate logit streams
            
            self.logAUROC(mode, logits, original_labels) 
            self.log_variance_health(mode, x_pert, x_recon_pert)

            self.log_pearson_metrics(mode, x_pert, x_ctrl, pert_idx)
            
        return total_loss, loss_class_critic


    def log_reconstruction(self,mode, x_pert, x_ctrl, x_recon_pert, x_recon_ctrl):
        batch_size = x_ctrl.size(0)
        n_valid_elements = self.gene_mask.sum() * batch_size

        diff_ctrl = (x_recon_ctrl - x_ctrl) * self.gene_mask
        diff_pert = (x_recon_pert - x_pert) * self.gene_mask
        loss_recon_ctrl = (diff_ctrl ** 2).sum() / n_valid_elements
        loss_recon_pert = (diff_pert ** 2).sum() / n_valid_elements
        loss_recon = loss_recon_ctrl + loss_recon_pert
        self.log(f"{mode}/Reconstruction_Loss", loss_recon, sync_dist=True,prog_bar=False)
        self.log(f"{mode}/Reconstruction_Pert_Loss", loss_recon_pert, sync_dist=True,prog_bar=False)
        self.log(f"{mode}/Reconstruction_Ctrl_Loss", loss_recon_ctrl, sync_dist=True,prog_bar=False)

        self.log_variance_health(mode, x_pert, x_recon_pert,postfix="_pert")
        self.log_variance_health(mode, x_ctrl, x_recon_ctrl,postfix="_ctrl")


    def log_nmse_top_genes(self, mode, x_pert, x_ctrl, pert_idx, top_n=20):
        """
        Calculates Normalized Mean Squared Error (NMSE) on the Top N 
        Differentially Expressed (DE) genes.
        """
        # 1. Generate Prediction
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            delta_pred = self.transition_fwd(z_ctrl, z_prompt)
            z_fake_pert = z_ctrl + delta_pred
            x_recon_pert = self.decoder(z_fake_pert)

        # 2. Calculate Deltas & Flatten [Seq, Dim] -> [Total_Genes]
        real_delta = (x_pert - x_ctrl).mean(dim=0).flatten()
        pred_delta = (x_recon_pert - x_ctrl).mean(dim=0).flatten()

        # 3. Identify Top N Movers
        top_vals, top_indices = torch.topk(torch.abs(real_delta), k=top_n)

        # 4. Extract values for these specific genes
        r_top = real_delta[top_indices]
        p_top = pred_delta[top_indices]

        # 5. Calculate MSE (The Error)
        mse = torch.mean((r_top - p_top) ** 2)

        # 6. Calculate Normalizer (The Magnitude of the real perturbation)
        # This represents the error if the model just predicted 0 (Identity)
        normalizer = torch.mean(r_top ** 2)

        # 7. Calculate NMSE
        # Add epsilon to prevent div by zero
        nmse = mse / (normalizer + 1e-8)

        # 8. Log it
        self.log(f"{mode}/NMSE_Top{top_n}", nmse, prog_bar=False, sync_dist=True)
        
        return nmse

    def log_pearson_metrics(self, mode, x_pert, x_ctrl, pert_idx):
        """
        Calculates Pearson Correlation per UNIQUE perturbation in the batch.
        Respects self.gene_mask to ignore padded genes.
        """
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            # Use flow matching solver instead of direct addition if applicable
            # z_fake_pert = self.ode_solve(z_ctrl, z_prompt, t_start=0, t_end=1, steps=4)
            # Or keep your current transition if you haven't switched yet:
            delta_pred = self.transition_fwd(z_ctrl, z_prompt)
            z_fake_pert = z_ctrl + delta_pred
            
            x_recon_pert = self.decoder(z_fake_pert)

        # 1. Identify unique perturbations in this batch
        unique_perts = torch.unique(pert_idx, dim=0)
        
        # Lists to store scores for this batch
        batch_pearson_all = []
        batch_pearson_top5 = []

        # Prepare the mask (flattened)
        # Assuming self.gene_mask is [1, Seq_Len, Dim] -> [1, Total_Genes]
        flat_mask = self.gene_mask.view(-1).bool()

        # 2. Iterate through each unique perturbation
        for p_id in unique_perts:
            if pert_idx.dim() > 1:
                mask = (pert_idx == p_id).all(dim=1)
            else:
                mask = (pert_idx == p_id)
            
            if mask.sum() == 0: continue

            # 3. Calculate Pseudo-bulk Vectors (Mean of this specific group)
            # Calculate mean across cells first [Batch_Subset, Genes] -> [Genes]
            real_delta_mean = (x_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()
            pred_delta_mean = (x_recon_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()
            
            # --- KEY CHANGE: Apply Gene Mask ---
            # We index into the flattened vector using the boolean mask
            real_delta_vector = real_delta_mean[flat_mask]
            pred_delta_vector = pred_delta_mean[flat_mask]
            
            # Safety: Check for flat vectors (std=0) to avoid NaNs
            if torch.std(real_delta_vector) < 1e-6 or torch.std(pred_delta_vector) < 1e-6:
                continue

            # --- Metric 1: Global Pearson ---
            p_all = pearson_corrcoef(pred_delta_vector, real_delta_vector)
            batch_pearson_all.append(p_all)

            # --- Metric 2: Top 5% Pearson ---
            # Calculate threshold on the REAL biological signal
            threshold = torch.quantile(torch.abs(real_delta_vector), 0.95)
            top_mask = torch.abs(real_delta_vector) > threshold
            
            if top_mask.sum() > 2:
                real_top = real_delta_vector[top_mask]
                pred_top = pred_delta_vector[top_mask]
                
                # Verify std again for the top subset
                if torch.std(real_top) > 1e-6 and torch.std(pred_top) > 1e-6:
                    p_top5 = pearson_corrcoef(pred_top, real_top)
                    batch_pearson_top5.append(p_top5)

        # 4. Aggregate and Log
        if len(batch_pearson_all) > 0:
            avg_pearson_all = torch.stack(batch_pearson_all).mean()
            self.log(f"{mode}/Pearson_All", avg_pearson_all, prog_bar=False, sync_dist=True)
        
        if len(batch_pearson_top5) > 0:
            avg_pearson_top5 = torch.stack(batch_pearson_top5).mean()
            self.log(f"{mode}/Pearson_Top5_Percent", avg_pearson_top5, prog_bar=False, sync_dist=True)


    def log_direction_error(self, mode, x_pert, x_ctrl, pert_idx, top_n=20):
        """
        Calculates Direction Error per UNIQUE perturbation in the batch.
        Prevents averaging different perturbations together.
        """
        # 1. Generate Prediction (Batch-wise for speed)
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            delta_pred = self.transition_fwd(z_ctrl, z_prompt)
            z_fake_pert = z_ctrl + delta_pred
            x_recon_pert = self.decoder(z_fake_pert)

        # 2. Identify unique perturbations
        unique_perts = torch.unique(pert_idx, dim=0)
        batch_errors = []

        # 3. Iterate through each perturbation group
        for p_id in unique_perts:
            mask = (pert_idx == p_id).all(dim=1)
            
            # Safety: Ensure we have cells for this perturbation
            if mask.sum() == 0: continue

            # 4. Calculate Pseudo-bulk Vectors for THIS perturbation only
            # We average the cells belonging to p_id
            real_delta = (x_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()
            pred_delta = (x_recon_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()

            # 5. Identify Top N Movers (Standard Logic)
            # Find genes with largest ABSOLUTE real change
            top_vals, top_indices = torch.topk(torch.abs(real_delta), k=top_n)

            # 6. Check Signs on those specific genes
            real_signs = torch.sign(real_delta[top_indices])
            pred_signs = torch.sign(pred_delta[top_indices])

            # 7. Calculate Error
            # Product is negative if signs are opposite (e.g. 1 * -1 = -1)
            sign_product = real_signs * pred_signs
            n_opposite = (sign_product < 0).float().sum()
            percent_opposite = n_opposite / float(top_n)
            
            batch_errors.append(percent_opposite)

        # 8. Aggregate and Log
        if len(batch_errors) > 0:
            # Average the error across all perturbations in this batch
            avg_error = torch.stack(batch_errors).mean()
            
            self.log(f"{mode}/Direction_Error_Top{top_n}", avg_error * 100.0, 
                    prog_bar=False, sync_dist=True)
            
            return avg_error.item()
        else:
            return 0.0



    def logAUROC(self, mode, logits, targets):
        """
        Dynamically logs AUROC and Confusion Matrix for any number of classes.
        Accepts raw logits and either 1D label indices or 2D one-hot targets.
        """
        # 1. Parse labels safely (handles both 1D and 2D target formats)
        if targets.dim() > 1 and targets.size(1) > 1:
            original_labels = torch.argmax(targets, dim=1)
        else:
            original_labels = targets.long()
            
        # 2. Convert to probabilities (AUROC metric requires probs in [0, 1])
        probs = torch.softmax(logits, dim=1)
            
        # 3. Build generic Multi-Hot Targets dynamically based on __init__
        B = original_labels.size(0)
        multi_hot_targets = torch.zeros((B, self.num_classes), dtype=torch.long, device=self.device)
        multi_hot_targets.scatter_(1, original_labels.unsqueeze(1), 1)
        
        # 4. Update Multi-Label AUROC (Accumulate only)
        self.classwise_auc.update(probs, multi_hot_targets)
        
        # 5. Update Confusion Matrix (Validation only)
        if mode == "val":
            if self.current_epoch % 5 == 0 and not self.trainer.sanity_checking:
                preds_discrete = torch.argmax(logits, dim=1)
                self.confmat.update(preds_discrete, original_labels)


    def logging_metrics(self, mode, lossDict):
        for key, value in lossDict.items():
            self.log(f'{mode}/{key}', value, prog_bar=True,sync_dist=True)
        
    def training_step(self, batch, batch_idx):
        if self.phase == 1:
            loss, classLoss = self.phase_one_shared_step(batch,mode="Train")
            main_opt, classifier_opt = self.optimizers()
            main_opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(main_opt,gradient_clip_val=0.5,gradient_clip_algorithm="norm")
            main_opt.step()
            classifier_opt.zero_grad()
            self.manual_backward(classLoss)
            classifier_opt.step()
            return loss
        
        else:
            opt_g, opt_d = self.optimizers()
            sched_g, sched_d = self.lr_schedulers()
            
            g_loss = self.shared_step_cycle_gan(batch, optimizer_idx=0, mode="Train")
            
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            self.clip_gradients(opt_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt_g.step()
            
            if batch_idx % 5 == 0:
                d_loss = self.shared_step_cycle_gan(batch, optimizer_idx=1, mode="Train")
                
                opt_d.zero_grad()
                self.manual_backward(d_loss)
                opt_d.step()
            
            # Step schedulers at the end of each epoch (last batch)
            if self.trainer.is_last_batch:
                sched_g.step()
                sched_d.step()
            return g_loss

    def validation_step(self, batch, batch_idx):
        if self.phase == 1:
            loss,class_loss = self.phase_one_shared_step(batch,mode="val")
            return loss + class_loss
        else:
            g_loss = self.shared_step_cycle_gan(batch, optimizer_idx=0, mode='val')
            d_loss = self.shared_step_cycle_gan(batch, optimizer_idx=1, mode='val')
            return g_loss + d_loss
        

    def configure_optimizers(self):
        classifier_params = list(self.latentClassifier.parameters())
        classifier_ids = list(map(id, classifier_params))

        main_params = [p for p in self.parameters() if id(p) not in classifier_ids]
        optimizer_main = torch.optim.AdamW(main_params, lr=1e-4, weight_decay=1e-3)
        optimizer_probe = torch.optim.AdamW(classifier_params, lr=1e-3, weight_decay=0.0)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, mode='min', factor=0.5, patience=5)
        #scheduler = LambdaLR(optimizer_main, lr_lambda=warmup_cosine_schedule)
        return [optimizer_main, optimizer_probe]

#        return [ {"optimizer": optimizer_main, "lr_scheduler": { "scheduler": scheduler,"monitor": "val/total_loss"}},{"optimizer": optimizer_probe}]

    
    def log_collapse_metrics2(self, mode, z_ctrl, z_fake_pert, z_real_pert):
        # Measure how far the model actually moved the points
        delta = z_fake_pert - z_ctrl
        delta_mag = torch.norm(delta, p=2, dim=1).mean()
        
        # Compare to how big the latent vectors are naturally
        z_mag = torch.norm(z_ctrl, p=2, dim=1).mean()
        movement_ratio = delta_mag / (z_mag + 1e-8)
        
        self.log(f"{mode}/Health_Delta_Mag", delta_mag, sync_dist=True)
        self.log(f"{mode}/Health_Movement_Ratio", movement_ratio, sync_dist=True)

        
        # Check the standard deviation of the OUTPUT batch.
        fake_std = z_fake_pert.std(dim=0).mean() 
        real_std = z_real_pert.std(dim=0).mean()
        
        # We want the Fake Diversity to match Real Diversity (Ratio ~ 1.0)
        diversity_ratio = fake_std / (real_std + 1e-8)
        
        self.log(f"{mode}/Health_Diversity_Real", real_std, sync_dist=True)
        self.log(f"{mode}/Health_Diversity_Fake", fake_std, sync_dist=True)
        self.log(f"{mode}/Health_Diversity_Ratio", diversity_ratio, sync_dist=True)

        delta_std = delta.std(dim=0).mean()
        self.log(f"{mode}/Health_Delta_Diversity", delta_std, sync_dist=True)


    def log_variance_health(self, mode, x_real, x_recon,postfix=""):
        """
        Checks if the model is capturing the dynamic range (variance) of the data
        or just predicting the mean (which has 0 variance).
        """
        # 1. Flatten Batch and Patch dimensions to get [Total_Cells, Genes]
        # We want variance across the entire batch/population
        # Assuming input is [Batch, Patches, Genes] -> [Batch*Patches, Genes]
        if x_real.dim() == 3:
            real_flat = x_real.reshape(-1, x_real.shape[-1])
            recon_flat = x_recon.reshape(-1, x_recon.shape[-1])
        else:
            real_flat = x_real
            recon_flat = x_recon

        # 2. Calculate Variance per Gene (dim=0)
        var_real = torch.var(real_flat, dim=0)
        var_recon = torch.var(recon_flat, dim=0)

        # 3. Identify the Top 50 Most Variable Genes in REAL data
        # (These are the ones that matter: Cell Cycle genes, CRISPR targets)
        top_vals, top_indices = torch.topk(var_real, k=50)

        # 4. Compare Variances ONLY on these Top 50 genes
        var_real_top = var_real[top_indices]
        var_recon_top = var_recon[top_indices]

        # 5. The Metric: Ratio of Reconstruction Variance to Real Variance
        # Avoid div by zero with 1e-8
        variance_preservation = var_recon_top.mean() / (var_real_top.mean() + 1e-8)

        # 6. Log it
        self.log(f"{mode}/Health_Var_Preservation{postfix}", variance_preservation, 
                prog_bar=False, sync_dist=True)
        
        # Optional: Log the Raw Variance numbers to see scale
        self.log(f"{mode}/Debug_Var_Real_Mean{postfix}", var_real_top.mean(), sync_dist=True)
        self.log(f"{mode}/Debug_Var_Recon_Mean{postfix}", var_recon_top.mean(), sync_dist=True)

        return variance_preservation
    
    def log_collapse_metrics(self, z, delta, z_cycle):
        # Calculate std across the batch
        z_std = z.std(dim=0) 
        # How many dimensions have significant variance?
        active_dims = (z_std > 0.01).sum().float()
        
        # Check for "Lazy" Transitions (Identity Collapse)
        delta_mag = delta.norm(dim=1).mean()
        z_mag = z.norm(dim=1).mean()
        delta_ratio = delta_mag / (z_mag + 1e-8)
        # Check Reversibility
        cycle_error = self.criterion(z, z_cycle)

        self.log(f"Health/Active_Dims", active_dims,sync_dist=True)      
        self.log(f"Health/Delta_Strength", delta_mag, sync_dist=True)     
        self.log(f"Health/Delta_to_Z_Ratio", delta_ratio, sync_dist=True)  
        self.log(f"Health/Z_Vector_Size", z_mag, sync_dist=True)          
        self.log(f"Health/Cycle_Integrity", cycle_error, sync_dist=True)  

    def on_validation_epoch_end(self):
        # We wrap in a try-except just in case DDP sanity checks run with empty batches
        if self.phase == 1:
            try:
                # 1. Compute the dictionary of class-wise AUROC scores
                auc_dict = self.classwise_auc.compute()
                
                # 2. Log the entire dictionary at once
                self.log_dict(auc_dict, sync_dist=True, prog_bar=False)
                
            except Exception as e:
                # Lightning sanity checks can sometimes cause empty metric computes
                print(f"Skipping AUROC compute on Rank {self.global_rank}: {e}")
                
            finally:
                # 3. Safely reset the metric for the next epoch
                self.classwise_auc.reset()

    