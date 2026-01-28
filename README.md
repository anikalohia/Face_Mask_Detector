# Face Mask Detector

A real-time **Face Mask Detector** using deep learning (CNN) and computer vision. This project detects whether a person is **wearing a mask** or **not wearing a mask** using webcam input.

---

## Features

- Real-time detection via webcam.
- Binary classification: `with_mask` vs `without_mask`.
- Trained using a convolutional neural network (CNN).
- Supports single image prediction.
- Lightweight and easy to integrate into other applications.

---

Model ==>

- Framework: TensorFlow / Keras

- Input: 224x224 images

- Architecture: CNN

- Output: Probability of with_mask and without_mask

- Trained model file: mask_detector_model.h5

- Dataset
Images of people with masks and without masks.

- Dataset split:

Training: 80%

Validation: 20%

- Dependencies: 
    Python 3.10+

    TensorFlow

    OpenCV

    NumPy

    Flask (optional, if using web interface)

pip install tensorflow opencv-python numpy flask