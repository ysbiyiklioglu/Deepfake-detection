# Deepfake Detection Project

This project aims to detect deepfake (fake) videos and images using deep learning models trained on the **FaceForensics++** and **Celeb-DF v2** datasets.

The project consists of three main components:

- A Streamlit-based demo application (`app.py`)
- A Jupyter notebook for data preprocessing and analysis (`main.ipynb`)
- A model evaluation script for Celeb-DF v2 (`eval_celebdf_v2.py`)

## Contents

- `app.py` → Web application allowing users to upload images or videos to test the model
- `main.ipynb` → Frame extraction, preprocessing, and analysis from the FaceForensics++ dataset
- `eval_celebdf_v2.py` → Comprehensive model evaluation on Celeb-DF v2
- `*.pth` files → Trained model checkpoints (should be placed in the project folder)

## Requirements

```bash
pip install torch torchvision torchaudio
pip install streamlit opencv-python pillow numpy matplotlib tqdm pandas scikit-learn
pip install facenet-pytorch  # Recommended for MTCNN face cropping
