# REGINA: Regularized Encoder with Latent Cycle-GAN for In-vitro Neural Cell Perturbation Approximation.
In this github we introduce every material needed for our REGINA research. This is a virtuall cell modelling pipeline with generative AI approaches centered around Cycle-GAN workflow.

## Installation
For the easies possible reproduction we included the singularity file we used to work with the pipeline. To be able to use it first you need to install singularity [link](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html).  

To install REGINA run:
``` 
singularity build --fix-perms REGINA.sif REGINA.def
``` 
## Data
Data is downloaded via GEARS [link](https://github.com/snap-stanford/GEARS):
``` 
from gears import PertData, GEARS
# get data
dataset_name = 'norman'
pert_data = PertData('./data')
# load dataset in paper: norman, adamson, dixit.
pert_data.load(data_name = f'{dataset_name}',data_path=None)
```
Our methods require to split the data into train-validation-test split. You can either use GEARS custom splitting method, which we used or use the split dict we got, and included.

## Preprocessing
Splitting the data:
```
train_adata = adata[adata.obs[perturbation_key].isin(custom_split_dict['train'])]
train_adata.write_h5ad("data/{dataset_name}/train.h5ad")
val_adata = adata[adata.obs[perturbation_key].isin(custom_split_dict['val'])]
val_adata.write_h5ad("data/{dataset_name}/val.h5ad")
test_adata = adata[adata.obs[perturbation_key].isin(custom_split_dict['test'])]
test_adata.write_h5ad("data/{dataset_name}/test.h5ad")

```

To use REGINA latent classifier you can add any given method to generate class information. The methods we used for dixit, norman, adamson dataset is:
``` 
def add_binary_state(adata):
    adata.var_names = adata.var.gene_name
    stress_prefixes = ('HSP', 'ATF', 'DNAJ', 'ERN', 'EIF2', 'CEBP')

    available_genes = adata.var_names.tolist()
    valid_markers = [g for g in available_genes if g.startswith(stress_prefixes)]
    sc.tl.score_genes(adata, gene_list=valid_markers, score_name='stress_score')
    threshold = adata.obs['stress_score'].quantile(0.70)
    adata.obs['cell_state'] = 'Homeostasis'
    adata.obs.loc[adata.obs['stress_score'] > threshold, 'cell_state'] = 'Stressed'

    print("\nFinal State Distribution:")
    print(adata.obs['cell_state'].value_counts())
    return adata
train_adata = ad.read_h5ad(f"data/{dataset_name}/train.h5ad")
val_adata = ad.read_h5ad(f"data/{dataset_name}/val.h5ad")
test_adata = ad.read_h5ad(f"data/{dataset_name}/test.h5ad")

train_adata = add_binary_state(train_adata)
val_adata = add_binary_state(val_adata)
test_adata = add_binary_state(test_adata)

train_adata.write_h5ad(f"data/{dataset_name}/train_processed.h5ad")
val_adata.write_h5ad(f"data/{dataset_name}/val_processed.h5ad")
test_adata.write_h5ad(f"data/{dataset_name}/test_processed.h5ad")
gene_to_idx = { gene:i for i, gene in enumerate(train_adata.var_names) }
import pickle
with open("data/{dataset_name}/gene_to_idx.pkl", "wb") as f:
    pickle.dump(gene_to_idx, f)
``` 


## Method

![Training pipeline of the regularized autoencoder.](figs/phase1.png)
*Figure 1: Training pipeline of the regularized autoencoder.*

### Phase One: Regularized Latent Autoencoder

Our approach follows the general paradigm of latent diffusion models, in which high-dimensional observations are first mapped into a lower-dimensional latent space via an autoencoder.

Given the high dimensionality of gene expression profiles, processing the full vector with standard linear layers is computationally heavy. To address this, we tokenize the input vector into a sequence of lower-dimensional segments. These segments are then processed by a Transformer encoder, utilizing bidirectional self-attention to capture complex, non-local gene dependencies.

Let $X \in \mathbb{R}^d$ denote a gene expression vector. An encoder $E_{\theta}$ projects $X$ into a latent representation $z$, which is subsequently reconstructed as $X_{rec}$ by a decoder $D_{\phi}$:

$$X_{rec}=D_{\phi}(E_{\theta}(X))$$

A standard autoencoder does not explicitly enforce preservation of biologically meaningful class information, such as the cellular state. To be able to study perturbation effects in the latent space, we augment the model with a latent classifier $C_{\Psi}$ that predicts the cell state directly from the latent representation, i.e., $\hat{c}_i=C_{\Psi}(z_i)$. We decided not to place it in the reconstruction space due to the high dimensionality of virtual cell modeling. In virtual cell analysis this cell state can be for example the cell cycle state, a population where the specific cell is originated, is it in stressed state.

#### Latent Space Regularization.
The classifier is trained using a stop-gradient operation on the encoder output, preventing its updates from affecting the encoder:

$$L_{sg}=CE(C_{\Psi}(sg[E_{\theta}(X)]),c)$$

where $sg[\cdot]$ denotes the stop-gradient operator. This loss is used exclusively for the classifier's training step.

Training a vanilla autoencoder typically results in an unregularized latent space that is unsuitable for downstream perturbation modeling. 
To address this, we introduce a *center loss* ($L_{CE}$) that encourages latent representations of the same biological state (e.g., control or perturbed) to form compact and well-separated clusters. 
We further employ an auxiliary regularization term ($L_{reg}$) to mitigate gradient instability during training. 

To additionally regularize the encoder, we include a second classification term without the stop-gradient:

$$L_{CLF}=CE(C_{\Psi}(E_{\theta}(X)),c)$$

This term encourages the encoder to produce latents that are themselves predictive of the cell state, while the classifier receives gradients from both terms (which only improves its accuracy and does not harm training dynamics).

The full autoencoder objective is then:

$$L_{AE}=\lambda_{MSE}L_{MSE}+\lambda_{CE}L_{CE}+\lambda_{reg}L_{reg}+\lambda_{CLF}L_{CLF}$$

where $L_{MSE}$ is the reconstruction loss, $L_{CE}$ the center loss, $L_{reg}$ combines auxiliary regularization terms, and the $\lambda$ coefficients are scalar hyperparameters.

#### Latent Consistency.
To ensure training stability and a self consistent encoder and decoder we applied consistency monitoring. A latent sample $z_i$ is decoded and re-encoded as $z_{rec}=E_{\theta}(D_{\phi}(z_i))$, and both representations are required to produce consistent class predictions:

$$L_{cons}=CE(C_{\Psi}(z_i),C_{\Psi}(z_{rec}))$$

where $CE(\cdot)$ denotes the cross-entropy loss. The training is stopped if this error starts to constantly increase.

### Second Phase: Latent Transition Modeling

![Training pipeline of the second phase of training.](figs/phase2_v3.png)
*Figure 2: Training pipeline of the second phase of training.*

Due to the absence of paired control–perturbation samples, a supervised latent transition model cannot be trained directly. We therefore adopt a *latent cycle GAN* framework that learns bidirectional mappings between control latent distributions $Z_{ctrl}$ and perturbed latent distributions $Z_{pert}$. 

A *forward transition block* $T^{fwd}_{\Theta_1}$ maps control latents to perturbed latents, while a *backward transition block* $T^{bwd}_{\Theta_2}$ performs the inverse mapping: 

$$Z_{pert}=T^{fwd}_{\Theta_1}(Z_{ctrl}), \qquad Z_{ctrl}=T^{bwd}_{\Theta_2}(Z_{pert})$$

Since the true inverse of a biological perturbation is unknown, the backward transition model is tasked with learning an implicit inversion corresponding to the perturbation that generated the observed perturbed state. Converting perturbed state to control state has no biological meaning, it was just a proxy task to increase the quality of our data generation.
To ensure consistency between these transformations, we apply a *cycle-consistency loss* $L_{cycle}$ that enforces the reconstruction of latent vectors after forward–backward and backward–forward passes. This loss utilizes MSE to minimize the distance between the original latents and their cycled counterparts:

$$L_{cycle}=MSE(Z_{ctrl},T^{bwd}_{\Theta_2}(T^{fwd}_{\Theta_1}(Z_{ctrl})))+MSE(Z_{pert},T^{fwd}_{\Theta_1}(T^{bwd}_{\Theta_2}(Z_{pert})))$$

#### Adversarial Training
Without additional constraints, the transition model may collapse to a trivial identity mapping, i.e., $T(Z)=Z$. To prevent this degeneration, we apply an adversarial discriminator $D_{\phi}$ operating in latent space, which distinguishes real latent vectors $Z_{real}$ from transformed vectors $\hat{Z}=T(Z)$. To ensure that the learned transformations are perturbation-specific, the discriminator is explicitly conditioned on the perturbation index $p$.
To stabilize training, $D_{\phi}$ is optimized using a Mean Squared Error (MSE) objective $L_{D}$. Here, the notation $Z_{ctrl,pert}$ refers to the input latent vector from either the control or perturbed domain, depending on the direction of the mapping:

$$L_{D}=MSE(D_{\phi}(Z_{real},p),1)+MSE(D_{\phi}(T(Z_{ctrl,pert}),p),0)$$

where $T(\cdot)$ denotes either the forward or backward transformation. 
The transition model is then trained using the corresponding generator loss $L_{G}$, which encourages the model to produce transformed latents that the discriminator classifies as "real" (target value of 1):

$$L_{G}=MSE(D_{\phi}(T(Z_{ctrl,pert}),p),1)$$

#### Prompt-Based Conditioning for Unseen Perturbations

To address unseen perturbations during training, we applied a *latent prompting* that explicitly encodes perturbation information. Given a perturbation index $p$, which denotes the index of the perturbed gene in the target gene expression vector, a synthetic gene expression vector $X_{fake}$ is constructed by copying the control expression and assigning a fixed value of $-1$ to the perturbed gene at index $p$. 
If multiple gene is perturbed the assigned value is also going to be multiplied with that amount. 
The corresponding latent embedding $Z_{fake}=E_{\theta}(X_{fake})$ is used to define a perturbation prompt
$Z_{prompt}=Z_{ctrl}-Z_{fake}$.

The transition model is conditioned on this prompt as:

$$Z^{fwd}_{pert}=T^{fwd}_{\Theta_1}(Z_{ctrl},Z_{prompt}), \qquad Z^{bwd}_{ctrl}=T^{bwd}_{\Theta_2}(Z_{pert},Z_{prompt})$$

The same prompt is used for both forward and backward transformations.
