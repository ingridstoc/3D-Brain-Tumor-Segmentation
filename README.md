## 3D Brain Tumors Segmentation and Classification using Deep Neural Networks 

## Description
This project implements a deep learning approach for brain tumor segmentation and classification using 3D MRI scans from the BraTS2021 (Brain Tumor Segmentation) dataset. By leveraging a 
3D U-Net architecture with a pre-trained VGG16 encoder, the model achieves accurate segmentation of different tumor regions, helping to advance medical image analysis
and potentially assist in clinical diagnosis and treatment planning.

## Dataset

```bibtex
@article{baid2021rsna,
  title={The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification},
  author={Baid, Upendra and Ghodasara, Satyam and Mohan, Suyash and Bilello, Michel and Calabrese, Evan and Colak, Errol and Farahani, Keyvan and Kalpathy-Cramer, Jayashree and Kitamura, Felipe C and Pati, Sarthak and others},
  journal={arXiv preprint arXiv:2107.02314},
  year={2021}
}
```
- Description: The dataset contains 3D MRI scans from 1,251 patients, with four imaging modalities per patient (T1, T1ce, T2, FLAIR) and corresponding tumor segmentation masks.

## Tumor Classification Classes
The model segments brain tumors into 4 distinct classes:

- **Class 0** : No tumor (healthy tissue)
- **Class 1**: NCR (Necrotic Tumor Core)
- **Class 2**: ED (Peritumoral Edematous/Invaded Tissue)
- **Class 3**: ET (Enhancing Tumor)

## Environment Setup

```bash
# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip3 install torch torchvision
pip3 install numpy
pip3 install matplotlib
pip3 install nibabel
pip3 install scikit-learn
```

## Implementation Details
- **Libraries Used**: `PyTorch`, `Numpy`, `Matplotlib`, `Nibabel`, `Scikit-learn`
- **Data Preprocessing**:
  + Data Splitting: Training: 70%, Validation: 20%, Testing: 10%
  + Normalization: Applied Z-score normalization to standardize the intensity values across all MRI scans.
  + Image Cropping: Reduced the depth dimension from 157 to 144 pixels to: eemove slices without tumor regions, reduce memory consumption, ensure compatibility with VGG16 (dimensions must be divisible by 16)
  + Data Organization: Organized data into 5 vectors: 4 vectors for each MRI modality (T1, T1ce, T2, FLAIR), 1 vector for ground truth segmentation masks
  + MRI Modalities

- **T1**: Standard T1-weighted imaging
- **T1ce**: T1-weighted with contrast enhancement
- **T2**: T2-weighted imaging
- **FLAIR**: Fluid-Attenuated Inversion Recovery

## Model Architecture
3D U-Net with VGG16 Encoder

- **Encoder**: Pre-trained VGG16 model adapted for 3D inputs
- **Decoder**: Transpose convolution layers for upsampling
- **Convolution Layers**: nn.Conv3d for 3D spatial feature extraction
- **Normalization**: BatchNormalization3D to prevent overfitting and maintain numerical stability
- **Activation Functions**:

+ ReLU after each BatchNormalization layer
+ Softmax at the output layer for pixel-wise classification

## Loss Function
Combined Loss: CrossEntropyLoss + DiceLoss

## Training Configuration

- **Optimizer**: Adam (typically used for medical image segmentation)
- **Evaluation Metrics**: Loss value, Dice Score per class

## Results

| Metric              | Value | 
|-------------------|----------|
| Final Loss | 0.2 |
| Dice Score (No Tumor)      |  100% | 
| Dice Score (Tumor Regions)          | 70% | 

## Conclusions

The 3D U-Net architecture with a pre-trained VGG16 encoder proves highly effective for brain tumor segmentation tasks.
The use of transfer learning (VGG16 encoder) combined with 3D convolutional layers allows the model to effectively capture both spatial features and cross-slice patterns in volumetric MRI data. This approach shows promise for assisting medical professionals in tumor diagnosis and treatment planning.

## Future Work

+ Add more neural networks architectures 
+ Implement data augmentation techniques
+ Fine-tune hyperparameters for improved tumor region segmentation
+ Evaluate on additional BraTS datasets for cross-validation
