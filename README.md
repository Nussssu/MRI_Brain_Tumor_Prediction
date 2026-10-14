<div align="center">

# 🧠 MRI Brain Tumor Prediction

**Multiclass brain tumor classification from MRI images using a hybrid DenseNet-121 + Transformer model, deployed as a Gradio web app.**

[![Hugging Face Space](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-FFD21E)](https://huggingface.co/spaces/Nussraat/MRI_brain_tumor_Prediction)
![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F97316?logo=gradio&logoColor=white)

**[🚀 Try the live demo →](https://huggingface.co/spaces/Nussraat/MRI_brain_tumor_Prediction)**

</div>

---

## Overview

This project presents a **multiclass brain tumor classification system** using MRI images and deep learning. The model is deployed as an interactive **web application** using **Gradio** and hosted on **Hugging Face Spaces**.

Users can upload an MRI image and instantly receive predicted probabilities for:
- Brain_Glioma
- Brain_Menin (Meningioma)
- Brain_Tumor

## Model Architecture

- **Backbone:** DenseNet-121 for convolutional feature extraction
- **Tokenization:** backbone feature maps projected into ViT-style tokens with a learnable `[CLS]` token
- **Encoder:** 6-layer Transformer encoder (12 heads, d_model = 768) with sinusoidal positional encoding
- **Head:** LayerNorm + linear classifier → softmax class probabilities
- **Preprocessing:** RGB conversion, resize, ImageNet normalization
- **Framework:** PyTorch

## Features

- Upload a brain MRI image through a simple web interface
- Instant top-3 class probabilities
- Runs on CPU or GPU automatically
- Publicly hosted demo on Hugging Face Spaces

## Tech Stack

| Purpose | Tools |
| --- | --- |
| Deep learning | PyTorch, Torchvision (DenseNet-121) |
| Web interface | Gradio |
| Image processing | Pillow, NumPy |
| Hosting | Hugging Face Spaces |

## Screenshots

### Web Interface
![Interface](Interface%20Screenshot/Interface_UI.png)

### Prediction: Meningioma
![Meningioma](Interface%20Screenshot/prediction_result_menin.png)

### Prediction: Tumor
![Tumor](Interface%20Screenshot/prediction_result_tumor.png)

### Prediction: Glioma
![Glioma](Interface%20Screenshot/prediction_result_glioma.png)

## Getting Started

```bash
git clone https://github.com/Nussssu/MRI_Brain_Tumor_Prediction.git
cd MRI_Brain_Tumor_Prediction
pip install -r requirements.txt
python app.py
```

> The trained weights (`best_model.pth`) are not stored in this repository because of their size. Download them from the [Hugging Face Space](https://huggingface.co/spaces/Nussraat/MRI_brain_tumor_Prediction/tree/main) and place the file next to `app.py`.

The app starts at http://localhost:7860.

## ⚠️ Disclaimer

This application is intended **for research and academic demonstration only** and should **not** be used for clinical diagnosis.

## Author

**Nusrat Jahan Ankhi** · Department of Computer Science · [GitHub](https://github.com/Nussssu) · [Hugging Face](https://huggingface.co/Nussraat)
