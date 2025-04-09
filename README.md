# Classification of Metals and Plastics using CNN

A novel deep learning approach for automating the segregation of metal and plastic waste using Convolutional Neural Networks (CNN). This project aims to assist recycling processes by accurately classifying input images of metal cans and plastic bottles, contributing to sustainable waste management.

## 🧠 Overview

This project presents an automated image classification model that can distinguish between **metal cans** and **plastic bottles**. Built with **Python** and **TensorFlow**, it utilizes **Convolutional Neural Networks (CNN)** for feature extraction and classification. The project was developed in response to the growing need for efficient waste segregation techniques, especially in recycling plants.

## 📌 Features

- Image classification using CNN  
- Self-made dataset with equal samples of metal and plastic  
- Preprocessing pipeline including resizing, grayscale conversion, and normalization  
- High classification accuracy (98.55%)  
- Lightweight, fast, and easy to deploy  

## 📂 Dataset

- Total Images: **6002**
  - **3001 Metal Can images**
  - **3001 Plastic Bottle images**
- Source: Self-collected from various image databases and real-world data
- Preprocessing:
  - Resized to **100x100 pixels**
  - Grayscale transformation
  - Normalization
  - Stored as pickle files: features and labels

## 🏗️ Architecture

- **2 Hidden Layers**: 64 neurons each, ReLU activation  
- **Max Pooling**: 2x2 for dimensionality reduction  
- **Output Layer**: 1 neuron with Sigmoid activation  
- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Evaluation Metrics**: Accuracy  

## 🧪 Model Training

- Framework: **TensorFlow / Keras**
- Model: **Sequential CNN**
- Batch Size: **300**
- Epochs: **10**
- Validation Split: **10%**

### Sample Results

| Epoch | Train Accuracy | Validation Accuracy |
|-------|----------------|---------------------|
| 1     | 90.98%         | 96.17%              |
| 5     | 96.81%         | 95.67%              |
| 10    | 98.52%         | 96.67%              |

- **Final Accuracy**: **98.55%**  
- **Final Loss**: **0.0522**

## 📦 Requirements

- Python 3.x  
- TensorFlow  
- NumPy  
- OpenCV  
- Matplotlib  
- Pickle  

Install the dependencies using:

```bash
pip install tensorflow numpy opencv-python matplotlib
```

## 🚀 Getting Started

1. Clone this repository:

```bash
git clone https://github.com/yourusername/metal-plastic-classifier.git
cd metal-plastic-classifier
```

2. Preprocess the dataset:

```python
python preprocess.py
```

3. Train the model:

```python
python train.py
```

4. Predict an image:

```python
python predict.py --image path_to_image.jpg
```

## 🎯 Applications

- Waste segregation in recycling plants  
- Smart bins with auto-classification  
- Educational tools for sustainability  
- Environmental monitoring systems  

## 📈 Future Enhancements

- Integrate real-time webcam input  
- Expand dataset for industrial-scale deployment  
- Incorporate transfer learning for improved accuracy  
- Build a mobile or web-based front-end interface  

## 📄 Research Publication

This project is based on the IEEE published research paper:  
📘 **[A Novel Approach for Classification of Metals and Plastics using CNN](https://ieeexplore.ieee.org/abstract/document/10307663)**  
Published in: *International Conference on Computing, Communication and Networking Technologies (ICCCNT), 2023*
