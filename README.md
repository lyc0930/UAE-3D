# Unified Variational Auto-Encoder for 3D Molecular Latent Diffusion Modeling (UAE-3D)

Official code implementation of [*Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling*](https://arxiv.org/pdf/2503.15567) (NeurIPS 2025).

## Environment Setup
```bash
conda create --name UAE-3D python=3.11
conda activate UAE-3D
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg pytorch-cluster pytorch-scatter -c pyg
conda install openbabel pandas transformers lightning peft fcd_torch rdkit -c conda-forge
conda install pomegranate moses -c conda-forge # for moses
# pip install wandb # for W&B
```

> [!WARNING]  
> conda-forge::moses is outdated and has bugs about `_mcf.append` in `moses/metrics/utils.py` and `from rdkit.six import iteritems` in `moses/metrics/SA_Score/sascorer.py`


## *De Novo* Generation
> [!TIP] 
> All the hyperparameters are set to the default values, and will be printed in the console in groups at the beginning of the training / sampling. The main parameters are listed in the following table.
> | Parameter | Value | 
> | --- | --- |
> | max_epochs | 2000 / 10000 (UAE / UDM) |
> | num_workers | 8 |
> | batch_size | 512 |
> | aug_rotation | True |
> | aug_translation | True |
> | aug_translation_scale | 0.1 |
> | learning_rate | 0.0001 |
> | weight_decay | 1e-05 |
> | encoder_hidden_dim | 64 |
> | encoder_n_heads | 8 |
> | encoder_blocks | 6 |
> | latent_dim | 16 |
> | decoder_hidden_dim | 64 |
> | decoder_n_heads | 8 |
> | decoder_blocks | 4 |
> | atom_loss_weight | 1.0 |
> | bond_loss_weight | 1.0 |
> | coordinate_loss_weight | 1.0 |
> | dist_loss_weight | 1.0 |
> | bond_dist_loss_weight | 10.0 |
> | kld_weight | 1e-08 |
> | diffusion_hidden_dim | 512 |
> | diffusion_n_heads | 8 |
> | diffusion_n_layers | 8 |
> | diffusion_mlp_ratio | 4.0 |
> | diffusion_dropout | 0.0 |
> | latent_whiten | isotropic |
> | noise_temperature | 0.95 |
> | noise_scheduler | cosine |
> | continuous_beta_0 | 0.1 |
> | continuous_beta_1 | 20 |
> | discrete_schedule | False |

### QM9 Training
> [!NOTE] 
> The QM9 dataset will be automatically downloaded and preprocessed in the first run.

```bash
python uae_trainer.py --filename='QM9/UAE' --devices="[0,]" --dataset='qm9' --root='data/QM9'
python udm_trainer.py --filename='QM9/UDM' --devices="[0,]" --dataset='qm9' --root='data/QM9' --vae_ckpt='./all_checkpoints/QM9/UAE/last.ckpt' --max_epochs=10000
```
### QM9 Sampling
```bash
python udm_trainer.py --test_only --filename='QM9/UDM' --devices="[0,]" --dataset='qm9' --root='data/QM9' --vae_ckpt='./all_checkpoints/QM9/UAE/last.ckpt'
```

### GEOM-Drugs Training
> [!NOTE] 
> To train on GEOM-Drugs, you may need to manually download the dataset from [GEOM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF) and put it in `data/GEOMDrugs`.

```bash
python uae_trainer.py --filename='GEOMDrugs/UAE' --devices="[0,]" --dataset='drugs' --root='data/GEOMDrugs'
python udm_trainer.py --filename='GEOMDrugs/UDM' --devices="[0,]" --dataset='drugs' --root='data/GEOMDrugs' --vae_ckpt='./all_checkpoints/QM9/UAE/last.ckpt' --max_epochs=10000
```
### GEOM-Drugs Sampling
```bash
python udm_trainer.py --test_only --filename='GEOMDrugs/UDM' --devices="[0,]" --dataset='drugs' --root='data/GEOMDrugs' --vae_ckpt='./all_checkpoints/GEOMDrugs/UAE/last.ckpt'
```

## Conditional Generation
### QM9 Training
> [!NOTE] 
> To conduct conditional generation, you may need to manually train property classifier networks following [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules?tab=readme-ov-file#train-a-property-classifier-network), and put it in `./data/QM9/property_classifier`.

```bash
python udm_trainer.py --filename='QM9/UDM' --devices="[0,]" --dataset='qm9' --root='data/QM9' --vae_ckpt='./all_checkpoints/QM9/UAE/last.ckpt' --max_epochs=10000 --condition_property='mu'
``` 

### QM9 Sampling
> [!NOTE]
> In the evaluation on conditional generation, the properties are randomly sampled from the distribution of the training set. To sample molecules with specific values, you may need to manually set the `context` of data batch.

```bash
python udm_trainer.py --test_only --filename='QM9/UDM' --devices="[0,]" --dataset='qm9' --root='data/QM9' --vae_ckpt='./all_checkpoints/QM9/UAE/last.ckpt' --condition_property='mu'
```


## Citation

If you use our codes or checkpoints, please cite your paper:

```bib
@inproceedings{luo2025towards,
    title        = {Towards Unified Latent Space for 3D Molecular Latent Diffusion Modeling},
    author       = {Yanchen Luo an Zhiyuan Liu and Yi Zhao and Sihang Li and Kenji Kawaguchi and Tat{-}Seng Chua and Xiang Wang},
    booktitle    = {The Thirty-Ninth Conference on Neural Information Processing Systems},
    year         = {2025},
    url          = {https://openreview.net/forum?id=g2XE40zTrj}
}
```