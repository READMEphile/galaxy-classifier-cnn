# Galaxy Classification Using Custom Convolutional Neural Networks

**Author:** Moulik Mishra  
**Institution:** Shiv Nadar University  
**Project Type:** Deep Learning in Astrophysics  
**Language:** Python  
**Status:** Completed

---

## Overview

This project implements a complete deep learning pipeline for galaxy classification using a custom-designed Convolutional Neural Network (CNN). The model is trained on a synthetic dataset that mimics the morphology of spiral, elliptical, and irregular galaxies. The pipeline includes data generation, augmentation, model training, evaluation, and explainability using Grad-CAM.

---

## Key Features

- **Synthetic Dataset Generator**: Generates 300 structured galaxy images representing spiral, elliptical, and irregular types.
- **Custom CNN Model**: Built using TensorFlow's functional API for clarity and flexibility.
- **Data Augmentation**: Includes random rotations, brightness adjustments, and horizontal flips.
- **Training Pipeline**:
  - ModelCheckpoint to save the best model
  - EarlyStopping to prevent overfitting
  - ReduceLROnPlateau for learning rate scheduling
- **Evaluation Metrics**:
  - Accuracy and loss curves
  - Confusion matrix
  - Classification report
- **Model Explainability**:
  - Grad-CAM visualizations of class-discriminative regions

---

## Project Structure

## Project Structure

- `galaxy_classification/`
  - `Galaxy.py` – Main Python script
  - `data/` – Synthetic image dataset
  - `models/` – Saved trained model
  - `results/` – Output plots and Grad-CAM images
  - `README.md` – Project documentation
 
## How to Run

1. Clone the repository and install dependencies (see `requirements.txt`).
2. Run the script:

   ```bash
   python Galaxy.py

3. The program will:

- Generate synthetic galaxy images  
- Train a CNN model  
- Evaluate its performance  
- Save training plots and Grad-CAM visualizations  

---

### Outputs

- `training_history.png`: Accuracy and loss across training epochs  
- `confusion_matrix.png`: Confusion matrix of predictions  
- `gradcam_overview.png`: Grid of Grad-CAM visualizations by class  
- `results/gradcam/`: Folder of individual Grad-CAM images  
- Trained model saved as `models/galaxy_cnn_best.h5`  

---

### Scientific Relevance

Galaxy morphology is closely linked to formation history and dynamical processes. Automating galaxy classification is critical for large surveys. This project demonstrates how custom deep learning models can simulate and classify galaxies in a scientifically interpretable and scalable way.

---

### Future Work

- Extend the model to classify real images from Galaxy Zoo.  
- Introduce more complex or hybrid architectures (e.g., residual or attention-based CNNs).  
- Deploy the classifier as a simple web tool with interactive Grad-CAM outputs.  

---

### Contact

**Email**: [mm748@snu.edu.in](mailto:mm748@snu.edu.in)  
**Affiliation**: Department of Physics, Shiv Nadar University  

