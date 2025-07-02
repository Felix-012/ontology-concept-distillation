# Project Setup Guide

## Prerequisites

* **PhysioNet credentialed account** to download the [MIMIC‑CXR‑JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) dataset.
* **UMLS license (UTS account)** to access the [UMLS Full Subset](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html).

## Dataset Downloads

Download and extract the datasets to locations of your choice.


> **Tip:** keep paths short and without spaces; you will reference them in the config file.

## Environment Setup

```bash
# clone and navigate to the repository
cd <path/to/your/repository>

# create and activate the conda environment
conda env create -n <envname> -f environment.yml
conda activate <envname>
```

## Configuration

Open `configs/config.yml` and replace the placeholder paths.

## Tasks

### 1. Retrieval Task

```bash
cd knowledge_graph_distance

python image_retrieval.py \
  --unbalanced_csv  ./splits/longtail_8_unbalanced_all.csv \
  --retrieval_csv   ./splits/longtail_8_balanced_retrieve.csv \
  --train_csv       ./splits/longtail_8_balanced_train.csv \
  --config_path     ./configs/config.yml \
  --output          /path/to/output.csv
```

### 2. Labeling Task

#### Stage 1 – Preliminary Labels

```bash
python label_csv.py \
  --src            ./splits/longtail_8_balanced_train.csv \
  --dst            /path/to/prelim_labels.csv \
  --mentions_path  /path/to/mentions.pkl
  --config_path    ./configs/config.yml
```

#### Stage 2 – Final Labels

```bash
python augment_labels.py \
  --csv1       ./splits/longtail_8_balanced_train.csv \
  --csv2       /path/to/prelim_labels.csv \
  --config     ./configs/config.yml \
  --mentions   /path/to/mentions.pkl
```








