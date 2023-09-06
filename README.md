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
python plot_perm.py --data_folder path/to/preprocessed/files --nshuffle N
```
for N different permuations of the input particles


To generate samples containing the likelihood information you can run

```bash
python plot_ll.py --sample [--ll] --nidx X
```
to sample from the model using MLE training (--ll) and for a given partition X of the data (the default option is to break the sample down into 200 slices. In this case, X should be a number between 0 and 199). This last number is used to generate samples in parallel where different jobs can be run at the same time for different values of the partition index.
After generating a few samples, you can merge the generated files using

```bash
python merge [--ll] --maxidx M
```
assuming you run M slices from 0 to M-1.

Plots can then be generated from the samples using
```bash
python plot_ll.py [--sup]
```
where the ```--sup``` flag loads the supervised classifier and compares with the direct density estimation methods.

