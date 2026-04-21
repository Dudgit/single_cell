from tqdm import tqdm
import torch
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.functional import pearson_corrcoef
import pandas as pd
if not hasattr(pd.Series, "nonzero"):
    pd.Series.nonzero = lambda self: self.to_numpy().nonzero()


class PrecisionEvaluator:
    def __init__(self, device='cuda'):
        self.device = device

    def pearson_corr(self, x_pert, x_ctrl, x_recon_pert, top_n=20):
        # 1. Pseudo-bulk: Average across cells (dim=0) to get 1D vectors (N_genes,)
        real_delta_vector = (x_pert.mean(dim=0) - x_ctrl.mean(dim=0))
        pred_delta_vector = (x_recon_pert.mean(dim=0) - x_ctrl.mean(dim=0))
        
        # 2. Find Top N genes
        top_vals, top_indices = torch.topk(torch.abs(real_delta_vector), k=top_n)

        # 3. Indexing now works perfectly because the vector is 1D
        real_top = real_delta_vector[top_indices]
        pred_top = pred_delta_vector[top_indices]
        
        pearson_all = pearson_corrcoef(pred_delta_vector, real_delta_vector)            
        pearson_top = pearson_corrcoef(pred_top, real_top)
        
        return pearson_top.item(), pearson_all.item() 

    def direction_error(self, x_pert, x_ctrl, x_recon_pert, top_n=20):
        # Pseudo-bulk
        real_delta = (x_pert.mean(dim=0) - x_ctrl.mean(dim=0))
        pred_delta = (x_recon_pert.mean(dim=0) - x_ctrl.mean(dim=0))
        
        top_vals, top_indices = torch.topk(torch.abs(real_delta), k=top_n)
        
        all_signs_real = torch.sign(real_delta)
        all_signs_pred = torch.sign(pred_delta)
        all_sign_product = all_signs_real * all_signs_pred
        n_opposite_all = (all_sign_product < 0).float().sum()
        percent_opposite_all = n_opposite_all / float(len(real_delta))

        real_signs = torch.sign(real_delta[top_indices])
        pred_signs = torch.sign(pred_delta[top_indices])
        sign_product = real_signs * pred_signs
        n_opposite = (sign_product < 0).float().sum()
        percent_opposite = n_opposite / float(top_n)
        
        return percent_opposite.item(), percent_opposite_all.item()

    def nmse_metric(self, x_pert, x_ctrl, x_recon_pert, top_n=20):
        # Pseudo-bulk
        real_delta = (x_pert.mean(dim=0) - x_ctrl.mean(dim=0))
        pred_delta = (x_recon_pert.mean(dim=0) - x_ctrl.mean(dim=0))
        
        mse_all = torch.mean((real_delta - pred_delta) ** 2)
        nmse_all = mse_all / torch.mean(real_delta ** 2)
        
        top_vals, top_indices = torch.topk(torch.abs(real_delta), k=top_n)
        r_top = real_delta[top_indices]
        p_top = pred_delta[top_indices]
        
        mse = torch.mean((r_top - p_top) ** 2)
        nmse = mse / torch.mean(r_top ** 2)
        
        return nmse.item(), nmse_all.item()
        
    def evaluate_batch(self, x_pert, x_ctrl, x_recon_pert, top_n=20):
        # Note: I fixed the typo person_corr -> pearson_corr here
        pearson_top5, pearson_all = self.pearson_corr(x_pert, x_ctrl, x_recon_pert, top_n)
        direction_error_top5, direction_error_all = self.direction_error(x_pert, x_ctrl, x_recon_pert, top_n)
        nmse_top5, nmse_all = self.nmse_metric(x_pert, x_ctrl, x_recon_pert, top_n)
        
        return pearson_top5, pearson_all, direction_error_top5, direction_error_all, nmse_top5, nmse_all

import torch
import numpy as np
from scipy.stats import wasserstein_distance, pearsonr

class DistributionEvaluator:
    def __init__(self, device='cuda'):
        self.device = device

    def compute_mmd(self, x_real, x_pred, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        Calculates Maximum Mean Discrepancy (MMD) using RBF Kernel.
        Low MMD = Distributions are similar.
        High MMD = Distributions are different.
        """
        n_samples = int(x_real.size(0))
        total = torch.cat([x_real, x_pred], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            with torch.no_grad():
                x = x_real
                dists = torch.cdist(x, x) ** 2
                bandwidth = torch.median(dists[dists > 0])
            
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        
        kernels = sum(kernel_val)
        
        XX = kernels[:n_samples, :n_samples]
        YY = kernels[n_samples:, n_samples:]
        XY = kernels[:n_samples, n_samples:]
        YX = kernels[n_samples:, :n_samples]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss.item()
    

    def compute_sliced_wasserstein(self,x_real,x_pred,num_projections=100):
        """
        Sliced Wasserstein Distance between joint distributions.
        Projects high-D gene expression onto random 1D directions.
        """

        # Move to CPU
        real = x_real.detach().cpu().numpy()
        pred = x_pred.detach().cpu().numpy()

        n_genes = real.shape[1]
        sw_dist = 0.0

        for _ in range(num_projections):
            # Random direction on unit sphere
            theta = np.random.normal(0, 1, size=(n_genes,))
            theta /= np.linalg.norm(theta) + 1e-8

            # Project
            real_proj = real @ theta
            pred_proj = pred @ theta

            # 1D Wasserstein
            sw_dist += wasserstein_distance(real_proj, pred_proj)

        return sw_dist / num_projections

    def compute_variance_preservation(self, x_real, x_pred):
        """
        Checks if the model captures the biological noise correctly.
        Calculates Pearson Correlation between Real SD and Predicted SD per gene.
        """
        # Calculate Standard Deviation per gene (axis 0 = cells)
        real_std = torch.std(x_real, dim=0).detach().cpu().numpy()
        pred_std = torch.std(x_pred, dim=0).detach().cpu().numpy()
        
        # Avoid NaNs if std is 0
        valid_idx = (real_std > 1e-6) & (pred_std > 1e-6)
        
        if valid_idx.sum() > 2:
            corr, _ = pearsonr(real_std[valid_idx], pred_std[valid_idx])
        else:
            corr = 0.0
            
        return corr
    
    def compute_l1_distance(self, x_real, x_pred):
        """
        Calculates Mean Absolute Error (L1) between the means of the populations.
        Good for checking if the 'center of mass' is correct.
        """
        # We compare the average expression profile of the batch
        real_mean = x_real.mean(dim=0)
        pred_mean = x_pred.mean(dim=0)
        
        l1_dist = torch.abs(real_mean - pred_mean).mean()
        return l1_dist.item()

    def compute_energy_loss(self, x_real, x_pred):
        """
        Calculates Energy Distance (Energy Statistic).
        D_E(X, Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
        This is a robust distance metric similar to MMD but using Euclidean norms.
        """
        n = x_real.size(0)
        m = x_pred.size(0)
        
        # Concatenate for efficient pairwise calculation
        total = torch.cat([x_real, x_pred], dim=0) # [2N, D]
        
        # Compute pairwise Euclidean distances using CDIST (More memory efficient than expand)
        # dists[i, j] = ||total[i] - total[j]||
        dists = torch.cdist(total, total, p=2) 
        
        # Extract sub-matrices
        # XX: Distances within Real
        dist_xx = dists[:n, :n].sum() / (n * n)
        
        # YY: Distances within Pred
        dist_yy = dists[n:, n:].sum() / (m * m)
        
        # XY: Distances between Real and Pred
        dist_xy = dists[:n, n:].sum() / (n * m)
        
        # Energy Distance Formula
        energy_loss = 2 * dist_xy - dist_xx - dist_yy
        return energy_loss.item()

    def evaluate_batch(self, x_real, x_pred):
        """
        Runs all metrics on a batch of cells.
        Expects inputs: [N_Cells, N_Genes]
        """
        return self.compute_mmd(x_real, x_pred),self.compute_sliced_wasserstein(x_real, x_pred), self.compute_variance_preservation(x_real, x_pred), self.compute_energy_loss(x_real, x_pred), self.compute_l1_distance(x_real, x_pred)