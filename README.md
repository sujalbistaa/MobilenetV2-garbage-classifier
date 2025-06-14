**MobileNetV2 Image Classification Model**

MobileNetV2 is a convolutional neural network architecture that is optimized for speed and low resource usage. This implementation is ideal for mobile applications or edge computing scenarios.

**This repository contains:**

Model architecture using MobileNetV2
Custom dataset support
Training and validation scripts
Model evaluation and visualization
Export to .h5, .tflite, or ONNX format (optional)


🧠 **Architecture**

Base Model: MobileNetV2 (pretrained on ImageNet)
Classifier Head: Fully connected layers adapted to your custom dataset
Activation: ReLU6, Softmax
Optimizer: Adam
Loss: Categorical Crossentropy / Binary Crossentropy
