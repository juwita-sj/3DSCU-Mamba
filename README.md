# 3DSCU-Mamba
Official repository for3D-SCUMamba: An Abdominal Tumor Segmentation model

<div align="center">


</div>

<p align="center" style="font-size: larger;">
  
</p>

![w:800px](GA_ieeeaccess.jpg)







## Dataset 
Dataset yang kami gunakan untuk mengevaluasi 3D-SCU mamba dapat diakses melalui link berikut 
1. MSD Pancreas dataset 
2. 

## Installation 
to do the installastion 

Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n 3dscu python=3.10 -y` and `conda activate 3dscu `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. Download code: `git clone https://github.com/juwita-sj/3DSCU`
5. `cd 3dscu/3dscu` and run `pip install -e .`


sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```



## Model Training
Download dataset from `data` folder. 3DSCU-Mamba is built on the popular [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework. If you want to train 3DSCU-Mamba on your own dataset, please follow this [guideline](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to prepare the dataset. 

### Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Train 3D models

- Train 2D `U-Mamba_Bot` model

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot
```

- Train 2D `U-Mamba_Enc` model

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaEnc
```

### Train 3D models

- Train 3D `U-Mamba_Bot` model

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaBot
```

- Train 3D `U-Mamba_Enc` model

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaEnc
```


## Inference

- Predict testing cases with `U-Mamba_Bot` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta
```

- Predict testing cases with `U-Mamba_Enc` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaEnc --disable_tta
```

> `CONFIGURATION` can be `2d` and `3d_fullres` for 2D and 3D models, respectively.

## Remarks

1. Path settings

The default data directory for U-Mamba is preset to U-Mamba/data. Users with existing nnUNet setups who wish to use alternative directories for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` can easily adjust these paths in umamba/nnunetv2/path.py to update your specific nnUNet data directory locations, as demonstrated below:

```python
# An example to set other data path,
base = '/home/user_name/Documents/U-Mamba/data'
nnUNet_raw = join(base, 'nnUNet_raw') # or change to os.environ.get('nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # or change to os.environ.get('nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results') # or change to os.environ.get('nnUNet_results')
```

2. AMP could lead to nan in the Mamba module. We also provide a trainer without AMP: https://github.com/bowang-lab/U-Mamba/blob/main/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEncNoAMP.py

## Paper

```
@article{U-Mamba,
    title={U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation},
    author={Ma, Jun and Li, Feifei and Wang, Bo},
    journal={arXiv preprint arXiv:2401.04722},
    year={2024}
}
```


## Acknowledgements

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [U-Mamba](https://github.com/state-spaces/mamba) for making their valuable code publicly available.






