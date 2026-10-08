# 🌍 GeoVision

### Multispectral Land-Use & Land-Cover Classification

> **Exploring deep learning for automated land-use and land-cover classification from satellite imagery.**

GeoVision is a computer vision project built around the **EuroSAT** dataset derived from Sentinel-2 satellite imagery. The project explores how convolutional neural networks can learn visual representations of different land-use and land-cover patterns and classify satellite images into distinct geographic categories.

The project was originally developed as an academic deep-learning study and is now presented as part of my evolving AI/ML portfolio.

---

## 🎯 Why This Problem Matters

Land-use and land-cover (LULC) classification is an important application of remote sensing and computer vision.

Automatically identifying surface categories from satellite imagery can support applications such as:

* 🌱 Agricultural monitoring
* 🌲 Forest and vegetation analysis
* 🏙️ Urban development
* 💧 Water-resource monitoring
* 🗺️ Geographic information systems
* 🌍 Environmental observation
* 📊 Land-resource management

Satellite imagery contains rich visual information, but different land-cover categories can have highly similar visual characteristics. This makes automated classification a meaningful computer-vision problem rather than a simple image-recognition task.

**GeoVision explores this problem using a deep residual convolutional network trained on the EuroSAT dataset.**

---

# 🛰️ Dataset

The project uses the **EuroSAT** dataset, a land-use and land-cover image dataset derived from Sentinel-2 satellite imagery.

| Property              | Details                          |
| --------------------- | -------------------------------- |
| Dataset               | EuroSAT                          |
| Satellite imagery     | Sentinel-2                       |
| Total labelled images | 27,000                           |
| LULC categories       | 10                               |
| Input resolution used | 64 × 64                          |
| Input channels        | 3                                |
| Task                  | Multi-class image classification |

### A note on multispectral imagery

EuroSAT originates from Sentinel-2 imagery containing multiple spectral bands. However, the **original implementation in this repository uses three-channel 64 × 64 image inputs**.

Therefore, this implementation should be understood as a **CNN-based satellite-image classification experiment using the visual representation available to the model**, rather than a full 13-band multispectral deep-learning pipeline.

This distinction is intentional and documented for reproducibility.

---

# 🧠 The Core Idea

The project investigates a simple question:

> **Can a deep residual CNN learn meaningful representations from satellite imagery well enough to distinguish different land-use and land-cover categories?**

The answer is explored through a complete deep-learning workflow:

```text
                  🛰️ EuroSAT Images
                         │
                         ▼
                Image Preprocessing
                         │
                  Resize → 64×64
                  Pixel Scaling
                         │
                         ▼
                  🧠 ResNet50
                         │
          ┌──────────────┴──────────────┐
          │                             │
    Convolutional Blocks          Identity Blocks
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
                  Learned Features
                         │
                         ▼
                  Average Pooling
                         │
                         ▼
                Fully Connected Layer
                         │
                         ▼
                  Softmax Output
                         │
                         ▼
              🌍 LULC Classification
```

---

# 🔬 Model Architecture

Rather than treating ResNet as a black-box API, the original implementation constructs a **ResNet50-style architecture from its fundamental building blocks**.

The network contains:

* Convolutional layers
* Batch normalization
* ReLU activations
* Identity residual blocks
* Convolutional residual blocks
* Shortcut connections
* Average pooling
* Fully connected classification layer
* Softmax output

### Residual Learning

A key idea behind ResNet is the use of **shortcut connections**.

Instead of forcing every layer to learn an entirely new transformation, residual blocks allow the network to learn a transformation relative to the information already flowing through the shortcut path.

Conceptually:

```text
Input
  │
  ├─────────────── Shortcut ──────────────┐
  │                                       │
  ▼                                       │
Conv → BN → ReLU → Conv → BN → Conv → BN │
  │                                       │
  └───────────────── Add ◄────────────────┘
                    │
                    ▼
                  ReLU
```

Implementing these blocks directly provided practical exposure to how a residual CNN is constructed internally.

---

# ⚙️ Training Configuration

| Component         | Configuration             |
| ----------------- | ------------------------- |
| Architecture      | Custom ResNet50-style CNN |
| Input size        | 64 × 64 × 3               |
| Number of classes | 10                        |
| Optimizer         | Adam                      |
| Loss function     | Categorical Cross-Entropy |
| Training epochs   | 20                        |
| Batch size        | 32                        |
| Pixel scaling     | 1 / 255                   |
| Validation split  | 20%                       |

---

# 📊 Model Evaluation

The project evaluates the classifier beyond a single accuracy value.

### Training behaviour

Training and validation curves are used to observe:

* Accuracy progression
* Loss progression
* Model convergence
* Differences between training and validation performance

### Confusion Matrix

A multi-class confusion matrix is used to understand how predictions are distributed across LULC categories.

This is particularly useful for identifying classes that are visually or spectrally difficult to distinguish.

```text
                 Predicted Classes
              ┌───────────────────────┐
              │                       │
Actual Classes│   Confusion Matrix    │
              │                       │
              └───────────────────────┘

      Correct predictions → Diagonal
      Misclassifications  → Off-diagonal
```

The original notebook contains the corresponding training curves and confusion-matrix analysis.

---

# 💡 What Makes the Project Interesting

### 01 — Real-world computer vision problem

The project is not based on a conventional object-recognition dataset. It works with **satellite imagery**, where visual patterns correspond to geographic land-cover characteristics.

### 02 — Deep residual learning

The project goes beyond a basic CNN by implementing the fundamental building blocks behind **ResNet-style residual learning**.

### 03 — End-to-end experimentation

The workflow covers:

**Data → preprocessing → model construction → training → validation → prediction → evaluation**

### 04 — Error analysis

The confusion matrix provides visibility into where the model succeeds and where different land-cover categories become difficult to distinguish.

### 05 — Remote sensing + AI

The project sits at the intersection of:

**Computer Vision + Deep Learning + Remote Sensing + Geospatial Applications**

---

# 🧪 What I Learned

Building GeoVision helped strengthen several foundational concepts in deep learning:

* Designing convolutional neural networks
* Understanding residual connections
* Working with image datasets
* Image preprocessing and normalization
* Multi-class classification
* One-hot encoded labels
* Categorical cross-entropy
* Batch normalization
* Model optimization with Adam
* Training/validation analysis
* Confusion-matrix based evaluation
* Interpreting model behaviour rather than relying only on accuracy

One of the most valuable aspects of the project was implementing the **ResNet building blocks manually**, which helped move the understanding of deep-learning architectures beyond simply importing a pre-built model.

---

# 🛠️ Technology Stack

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras
* Convolutional Neural Networks
* ResNet-style architecture

### Machine Learning

* Scikit-learn

### Data & Computation

* NumPy

### Visualization

* Matplotlib
* Seaborn

### Domain

* Computer Vision
* Remote Sensing
* Land-Use / Land-Cover Classification

---

# 📈 Project Results

The original implementation produced **strong classification performance** on the EuroSAT validation data.

Model behaviour was analysed using:

**Training Accuracy → Validation Accuracy → Training Loss → Validation Loss → Confusion Matrix**

The evaluation demonstrated that the residual CNN was able to learn meaningful visual representations for the multi-class LULC classification task.

> **Note:** Numerical results are intentionally kept tied to the original experiment outputs rather than introducing reconstructed or estimated values into this README.

---

# 🔎 From Academic Project to Portfolio Project

GeoVision began as an academic deep-learning project.

The original implementation and project report are preserved rather than rewritten to imply capabilities that were not part of the original experiment.

The focus of this portfolio version is to make the work easier to understand, reproduce, evaluate, and discuss from an engineering perspective.

This distinction matters:

> **The project represents what was actually implemented, while the future-work section represents how the system could be evolved today.**

---

# 🚀 If I Extended GeoVision Today

There are several natural directions for taking the project further:

### 🛰️ Full multispectral modelling

Move beyond the original three-channel input and investigate the additional Sentinel-2 spectral information available in EuroSAT.

### 🧠 Model benchmarking

Compare the custom ResNet implementation against alternative CNN architectures and transfer-learning approaches.

### 📊 Deeper evaluation

Add:

* Precision
* Recall
* F1-score
* Per-class performance
* ROC/PR analysis where appropriate
* More detailed error analysis

### 🔬 Explainable AI

Investigate which image regions contribute most strongly to individual predictions using techniques such as Grad-CAM.

### 🌍 Spatial validation

Explore validation strategies that account for spatial relationships within remote-sensing data rather than relying only on conventional random splits.

### ⚡ Deployment

Package the trained model into an inference service capable of accepting an image and returning the predicted land-cover category.

These are **future directions**, not claims about the original implementation.

---

# 📚 Project Background

Land-use and land-cover classification is widely studied in remote sensing because changes in Earth's surface can provide important information for environmental monitoring, natural-resource management, urban planning, and geographic analysis.

The original academic report accompanying this project contains the literature background, problem formulation, dataset discussion, methodology and conclusions.

The report is preserved in the repository for complete project context.

---

# 📁 Repository

```text
GeoVision-Multispectral-Land-Use-Classification/
│
└── Land Use Land Cover Classification Using MultiSpectral Images/
    │
    ├── Python implementation
    └── Academic project report (PDF)
```

---

# 📄 Original Project Report

The complete academic report is available in this repository:

**`SOFT_COMP_LAND_USE_LAND_COVER_CLASSIFICATION_USING_MULTISPEC.pdf`**

It provides the original academic context and documentation behind the project.

---

# 🎓 Project Context

**Academic Project — Computer Vision / Machine Learning**

The project represents an early hands-on exploration of applying deep learning to a real-world remote-sensing problem.

Revisiting it as part of my professional portfolio reflects an ongoing progression from **academic experimentation toward stronger AI/ML engineering practices and technical communication.**

---

## 🌍 GeoVision in One Sentence

> **A deep-learning computer vision project that explores automated land-use and land-cover classification from Sentinel-2-derived satellite imagery using a custom ResNet50-style convolutional architecture.**

---
