<img src="https://github.com/QinZhou-work/DeSpaST/blob/36c54578fd902fe8f0b4c8a19bcbcb0268089a08/figures/qbrc_logo.png?inline=false" width="40%"/>


# DeSpaST

Deconvoluting Spatial Transcriptomics Spot-level Signals to Single-Cell Resolution Using Histology Images

# Introduction
This script implements DeSpaST, a model that integrates histology-derived cell features with spatial transcriptomics data using dynamic edge-conditioned graph convolution. It segments cells, extracts morphological and spatial features, and builds cellular graphs to deconvolute spot-level expression into single-cell profiles, improving spatial resolution and interpretability.

<div align="center">
  <img src="https://github.com/QinZhou-work/DeSpaST/blob/36c54578fd902fe8f0b4c8a19bcbcb0268089a08/figures/figure1_github.png?inline=True" alt="Alt text" width="90%"/>
</div>

# Installation
DeSpaST works with Python 3.6.6. Please make sure you have the correct verion of Python.  

```
conda create -n "DeSpaST" python=3.6.6
conda activate DeSpaST
pip install -r DeSpaST_requirements.txt
```

# User Guideline
### a) Spatial transcriptmoic data processing

- [ ] [Spatial transcriptmoic data processing](https://git.biohpc.swmed.edu/QBRC/deep-learning/development/DESTINY/-/settings/integrations)

### b) Image feature extraction:

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

### c) Constructe the model 



### d) Deconvolved gene expression for each cell

