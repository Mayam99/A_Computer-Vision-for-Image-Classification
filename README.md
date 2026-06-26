# 📸 Natural Image Classification System
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)](https://share.streamlit.io/)

A production-grade Deep Learning application that classifies images into 8 distinct categories with **99.71% accuracy**. Built using Transfer Learning (VGG16) and deployed as a live web application.

---

## 🚀 Live Demo
**[Click here to view the Live App on Streamlit Cloud]((https://acomputer-vision-for-image-classification-948o6prkixb5dhx8zkjv.streamlit.app/))**

---

## 📌 Project Overview
This project addresses the challenge of multi-class image recognition across diverse environmental domains. It compares a custom baseline CNN with an advanced Transfer Learning approach to achieve state-of-the-art performance.

### 8 Target Classes:
`airplane`, `car`, `cat`, `dog`, `flower`, `fruit`, `motorbike`, `person`

---

## 📊 Performance Showdown
The champion model (VGG16) decimated the baseline custom CNN in both accuracy and computational efficiency.

| Metric | Custom CNN (Baseline) | VGG16 (Transfer Learning) |
| :--- | :---: | :---: |
| **Final Accuracy** | 85.60% | **99.71%** |
| **Validation Loss** | 0.3840 | **0.0153** |
| **Training Time/Epoch** | 238 Seconds | **36 Seconds** |
| **Convergence** | 30+ Epochs | **15 Epochs** |

---

## 🛠️ Advanced Engineering Solutions
This project features several production-grade workarounds to ensure stability in cloud environments:

1. **Cloud-Streaming Model Engine:** Since the model asset (56MB) exceeds GitHub's browser upload limit, the app uses `gdown` to dynamically stream and cache the model from a secure Google Drive repository on its first boot.
2. **Metadata Monkey-Patching:** Resolved a critical Keras version mismatch error (`quantization_config`) by implementing a runtime patch that intercepts layer initialization and strips conflicting metadata before model deserialization.
3. **Execution Guarding:** Implemented a recursion-guard flag to prevent infinite loops during Streamlit page re-runs.

---

## 💻 Tech Stack
*   **Deep Learning:** TensorFlow, Keras v3
*   **Architecture:** VGG16 (Fine-tuned)
*   **UI/Frontend:** Streamlit
*   **Environment:** Anaconda, VS Code, Google Colab
*   **Libraries:** NumPy, Pillow, Gdown

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/Mayam99/YOUR_REPO_NAME.git](https://github.com/Mayam99/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
