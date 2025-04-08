# Drug-Drug Interaction Prediction Model (DDIMDL)

This project implements a multi-modal deep learning model for predicting drug-drug interactions (DDIs) using chemical, protein, enzyme, and pathway information.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── model.py              # DDIMDL model implementation
├── generate_dummy_data.py # Dummy data generation for testing
└── train.py             # Training and evaluation scripts
```

## Features

- Multi-modal drug feature encoding:
  - Chemical structure features using RDKit
  - Protein interaction networks using Graph Neural Networks
  - Enzyme information encoding
  - Pathway sequence encoding using LSTM
- Binary classification for drug-drug interactions
- Evaluation metrics including AUC-ROC and Loss curves
- Training visualization with matplotlib

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Generate dummy dataset and train the model:

```bash
python train.py
```

This will:
- Generate a dummy dataset of 100 drugs
- Split the data into train/validation/test sets
- Train the DDIMDL model for 50 epochs
- Save training metrics visualization
- Save the trained model

2. Model Architecture

The DDIMDL model consists of four main components:
- ChemicalEncoder: Processes molecular features
- ProteinEncoder: Graph Neural Network for protein interaction networks
- EnzymeEncoder: Processes enzyme information
- PathwayEncoder: LSTM network for pathway sequences

3. Output

The training script will output:
- Training and validation loss at each epoch
- Validation AUC scores
- Final test set performance metrics
- A plot of training metrics saved as 'training_metrics.png'
- Trained model weights saved as 'ddimdl_model.pth'

## Model Parameters

Current default parameters:
- Hidden dimension: 64
- Learning rate: 0.001
- Batch size: 32
- Number of epochs: 50

## Note

This implementation uses dummy data for demonstration. For real applications:
1. Replace the dummy data generation with actual drug data from DrugBank
2. Adjust model parameters based on your dataset
3. Implement additional data preprocessing as needed
4. Consider adding more sophisticated feature extraction methods

## Future Improvements

1. Add support for more molecular descriptors
2. Implement attention mechanisms in the encoders
3. Add more sophisticated protein feature extraction
4. Implement cross-validation
5. Add support for different types of drug-drug interactions
6. Implement interpretability methods 