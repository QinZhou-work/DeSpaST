<img src="https://github.com/QinZhou-work/DeSpaST/blob/36c54578fd902fe8f0b4c8a19bcbcb0268089a08/figures/qbrc_logo.png?inline=false" width="40%"/>


# DeSpaST

Deconvoluting Spatial Transcriptomics Spot-level Signals to Single-Cell Resolution Using Histology Images

Author: Qin Zhou (qin.zhou@utsouthwestern.edu)

# Introduction
This script implements DeSpaST, a model that integrates histology-derived cell features with spatial transcriptomics data using dynamic edge-conditioned graph convolution. It segments cells, extracts morphological and spatial features, and builds cellular graphs to deconvolute spot-level expression into single-cell profiles, improving spatial resolution and interpretability.

<div align="center">
  <img src="https://github.com/QinZhou-work/DeSpaST/blob/36c54578fd902fe8f0b4c8a19bcbcb0268089a08/figures/figure1_github.png?inline=True" alt="Alt text" width="90%"/>
</div>

# Installation
The core dependencies for this project contains

- torch
- torchvision
- spicy
- numpy
- pandas
- opencv-python
- Pillow
- matplotlib
- scikit-image
- openslide
- PyYAML

Install the dependencies by pip install requirements

```
conda create -n "DeSpaST" python=3.6.6
conda activate DeSpaST
pip install -r DeSpaST_requirements.txt
```

Install 'openslde' by the conda install
```
conda install -c conda-forge openslide
```
### [Troubleshooting]

If you cann't install 'opencv-python' and get the error message like "ERROR: Could not build wheels for opencv-python which use PEP 517 and cannot be installed directly", you can fix it by

```
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade cmake
python -m pip install --prefer-binary opencv-python
```

# User Guideline

### Step 1. Image feature extraction:

Cells in the images can be segmented using our methods or other similiar method.
<pre>
python extract_image_feature.py \
  --input-dir ./example_data \
  --output-dir ./output/ \
  --model-dir ./models/image/ \
  --slide-id CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_tissue_image \
  --device cuda:0
</pre>

For more reference:

HD-Staining: https://lce.biohpc.swmed.edu/maskrcnn/

HD-YOLO: https://github.com/impromptuRong/hd_wsi

After segmenting cells from H&E images, the extracted cell information can then be input into the model.

### Step 2. Constructe the model 

[Constructe the model](https://github.com/QinZhou-work/DeSpaST/blob/dd54a198cd9e0233d07c396c3413a15aa0aa9725/tutorial/DespaST_1_model_construction.ipynb)

### Step 3. Inference gene expression for each cell

[Resolution enhancement](https://github.com/QinZhou-work/DeSpaST/blob/dd54a198cd9e0233d07c396c3413a15aa0aa9725/tutorial/DespaST_2_Inference.ipynb)
