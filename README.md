# Handwritten Digit Generator using GAN

## Overview
This project implements a **Generative Adversarial Network (GAN)** to generate realistic handwritten digits resembling those in the MNIST dataset. The GAN consists of two neural networks: a **Generator** that creates fake digit images from random noise and a **Discriminator** that distinguishes between real and fake images. The adversarial training process improves both models, resulting in high-quality generated digits. The implementation uses PyTorch and is designed to run on a GPU (CUDA) for faster training.

## Requirements
To run this project, ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- torchsummary
- tqdm

Install dependencies using:
```bash
pip install torch torchvision numpy matplotlib torchsummary tqdm
```

## Dataset
The project uses the **MNIST dataset**, which includes 60,000 training images of handwritten digits (0–9), each of size 28x28 pixels in grayscale. The dataset is automatically downloaded via `torchvision`.

## Model Architecture

### Generator
- **Input**: Random noise vector (dimension: 64)
- **Architecture**:
  - ConvTranspose2d layers to upsample from noise to 28x28x1 images
  - ReLU activations with BatchNorm (except for the final layer)
  - Final layer: ConvTranspose2d with ReLU activation
- **Output**: 28x28x1 grayscale image

### Discriminator
- **Input**: 28x28x1 image (real or generated)
- **Architecture**:
  - Conv2d layers with LeakyReLU activations and BatchNorm
  - Final layer: Linear layer outputting a single value (real/fake probability)
- **Output**: Probability that the input is real

## Training
- **Batch Size**: 128
- **Epochs**: 20
- **Learning Rate**: 0.0002
- **Optimizer**: Adam (beta_1=0.5, beta_2=0.99)
- **Loss Function**: Binary Cross-Entropy with Logits
- **Device**: CUDA (GPU) if available, otherwise CPU

The training alternates between:
1. Updating the Discriminator with real (MNIST) and fake (generated) images.
2. Updating the Generator to produce more convincing images.

## Code Structure
The project is implemented in a Python script or Jupyter Notebook with the following components:

1. **Data Loading**:
   - Loads MNIST dataset with random rotation augmentation.
   - Batches data using `DataLoader`.

2. **Model Definition**:
   - Generator: Upsamples noise to 28x28 images.
   - Discriminator: Classifies images as real or fake.

3. **Weight Initialization**:
   - Uses normal distribution (mean=0, std=0.02) for Conv2d, ConvTranspose2d, and BatchNorm layers.

4. **Training Loop**:
   - Iterates over epochs, updating Discriminator and Generator.
   - Tracks and prints average losses per epoch.

5. **Visualization**:
   - Displays generated images using Matplotlib.

## Usage
1. Clone the repository or download the code.
2. Install dependencies (see Requirements).
3. Run the script or notebook:
   ```bash
   python gan_mnist.py
   ```
   or open and execute the cells in a Jupyter Notebook.
4. Monitor training progress via printed loss values.
5. After training, view generated digits using the visualization function.

## Example Output
After training for 20 epochs, the Generator produces images resembling MNIST digits. Generated images are displayed in a grid using Matplotlib.

## Notes
- Training GANs can be unstable; adjust hyperparameters (e.g., learning rate, epochs) if needed.
- Use a GPU for faster training (set `device='cuda'`).
- The quality of generated digits improves with more epochs.
- For better results, experiment with different noise dimensions or network architectures.

## References
- Blog Post: [Generative Adversarial Networks (GANs): The AI Behind Realistic Fake Data](https://example.com)
- PyTorch Documentation: [pytorch.org](https://pytorch.org)
- MNIST Dataset: [torchvision.datasets.MNIST](https://pytorch.org/vision/stable/datasets.html)
