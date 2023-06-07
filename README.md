# High-dimensional and Permutation Invariant Anomaly Detection

This repository contains all the scripts used to derive the results presented in the paper "High-dimensional and Permutation Invariant Anomaly Detection".

## Data
We use the [Top Quark Tagging Reference Dataset](https://zenodo.org/record/2603256) and a Z' dataset that will be available soon.

To preprocess the top tagging data you can run:

```bash
python preprocess.py --folder path/to/saved/file
```

## Training

To train the diffusion model you can run:

```bash
python train.py --dataset [top_tagging/gluon_tagging] [--ll] [--sup] --data_path path/to/preprocessed/files
```

where the flag ```--ll``` trains the model using MLE and ```--sup``` instead trains the supervised classifier.

## Evaluation

To test the permutation invariance of the likelihood estimation you can run:

```bash
python plot_perm.py --data_folder path/to/preprocessed/files
```

To plot the SIC, ROC, and the likelihood distributions you can run

```bash
python plot_ll.py [--sup]
```
where the ```--sup``` flag loads the supervised classifier and compares with the direct density estimation.

