# Face Recognition Trainer

This repository contains code to train a face recognition model using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm. The model is trained on images of faces organized by person in subdirectories and generates a `.yml` file with the trained model, along with the features and labels saved as `.npy` files.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)

## Requirements

- Python 3.x
- OpenCV with `opencv-contrib-python` package for extended modules
- NumPy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/anubhab7111/face-recognition-trainer.git
   cd face-recognition-trainer
   ```

2. Install the required Python packages:
   ```bash
   pip install opencv-contrib-python numpy
   ```

3. Ensure you have a trained Haar Cascade XML file for face detection:
   - You can download it [here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and place it in the project root or specify its path.

## Project Structure

```plaintext
face-recognition-trainer/
├── haar_face.xml                # Haar Cascade XML file for face detection
├── Resources/
│   └── Faces/
│       └── train/
│           ├── person1/         # Folder for person 1's images
│           ├── person2/         # Folder for person 2's images
│           └── ...              # More folders for each person
├── train_faces.py               # Main script for training the face recognition model
├── face_trained.yml             # Trained model file (generated after running the script)
├── features.npy                 # Saved features (generated after running the script)
└── labels.npy                   # Saved labels (generated after running the script)
```

## Usage

1. **Organize Training Data**:
   - Inside `Resources/Faces/train/`, create a folder for each person with their name as the folder name.
   - Add images of each person to their respective folder.

2. **Run the Training Script**:
   ```bash
   python train_faces.py
   ```
   - This will create the `face_trained.yml`, `features.npy`, and `labels.npy` files in the project directory.

3. **Model Output**:
   - `face_trained.yml` – The trained model file.
   - `features.npy` and `labels.npy` – Saved features and labels for future use or testing.
