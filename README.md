# Computer Vision and Image Processing:

---

## 📌 Project Overview
This project demonstrates key concepts in Computer Vision, ranging from classical image processing (Edge Detection) to feature matching (SIFT/ORB) and modern Deep Learning classification (CNN vs. SVM).

---

## 📂 Project Tasks

### Task 1: Canny Edge Detection
Implementation of the multi-stage Canny algorithm to identify structural boundaries.
* **Process:** Noise Reduction (Gaussian Blur) → Gradient Calculation (Sobel) → Non-Maximum Suppression → Hysteresis Thresholding.
* **Key Feature:** Demonstration of threshold adjustments to optimize edge-map clarity.

* **Sample:**
![Canny Edge Detection Result](image.png)
### Task 2: Feature Matching (SIFT & ORB)
Comparison of local feature identification and matching.
* **SIFT (Scale-Invariant Feature Transform):** High-precision gradient-based matching.
* **Sample for SIFT algorithm**
![SIFT algorithm Result](sift.jpg)
* **ORB (Oriented FAST and Rotated BRIEF):** High-speed, 
efficient alternative to SURF/SIFT.
* **Sample for SIFT algorithm**
![SURF algorithm Result](surf.jpg)

* **Metric:** Uses Brute-Force (BF) Matcher with $L2$ Norm for SIFT and Hamming Distance for ORB.

### Task 3: Image Classification (Cat vs. Dog)
A benchmarking study between Classical Machine Learning and Deep Learning.
* **Classical Model:** SVM (Support Vector Machine) with RBF Kernel and flattened pixel features.
* **Deep Learning Model:** CNN (Convolutional Neural Network) using spatial hierarchies.
* **Optimizations:** Adam Optimizer, Dropout for regularization, and Pixel Normalization.
* **Sample**
![CNN vs SVM Result](cnnvssvm.jpg)

---

## 🛠️ Technologies & Libraries
* **Python 3.x**
* **OpenCV:** Image processing and feature detection.
* **TensorFlow/Keras:** Deep Learning model architecture.
* **Scikit-Learn:** SVM implementation and performance metrics.
* **Matplotlib & Seaborn:** Results visualization and confusion matrices.

---

## 🚀 How to Run (Google Colab)

1. **Environment Setup:** Ensure you have a GPU runtime enabled for Task 3 (**Runtime > Change runtime type > GPU**).
   
2. **Directory Structure:**
   Create the following folders in the sidebar:
   ```bash
   dataset/
   ├── Cat/
   └── Dog/