import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

def generate_dummy_smiles(n_samples=100):
    """Generate dummy SMILES strings."""
    # Simple SMILES patterns for demonstration
    patterns = ['CC', 'CCO', 'CCCC', 'c1ccccc1', 'CC(=O)O', 'CCN']
    return np.random.choice(patterns, size=n_samples)

def extract_chemical_features(smiles):
    """Extract chemical features from SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ]
    except:
        return [0, 0, 0, 0, 0]

def generate_dummy_dataset(n_drugs=100):
    """Generate a complete dummy dataset."""
    # Generate chemical features
    smiles = generate_dummy_smiles(n_drugs)
    chemical_features = np.array([extract_chemical_features(s) for s in smiles])
    
    # Generate protein features (dummy graph data)
    protein_features = []
    protein_edge_indices = []
    for _ in range(n_drugs):
        n_nodes = np.random.randint(5, 10)
        features = torch.randn(n_nodes, 10)  # 10-dimensional node features
        edges = torch.randint(0, n_nodes, (2, n_nodes * 2))  # Random edges
        protein_features.append(features)
        protein_edge_indices.append(edges)
    
    # Generate enzyme features (one-hot encoded)
    n_enzyme_types = 5
    enzyme_features = np.random.randint(0, 2, (n_drugs, n_enzyme_types))
    
    # Generate pathway features (sequence data)
    pathway_length = 10
    pathway_dim = 8
    pathway_features = np.random.randn(n_drugs, pathway_length, pathway_dim)
    
    # Generate interaction matrix (symmetric)
    interactions = np.zeros((n_drugs, n_drugs))
    for i in range(n_drugs):
        for j in range(i+1, n_drugs):
            # 20% chance of interaction
            if np.random.random() < 0.2:
                interactions[i,j] = interactions[j,i] = 1
    
    return {
        'chemical_features': torch.FloatTensor(chemical_features),
        'protein_features': protein_features,
        'protein_edge_indices': protein_edge_indices,
        'enzyme_features': torch.FloatTensor(enzyme_features),
        'pathway_features': torch.FloatTensor(pathway_features),
        'interactions': torch.FloatTensor(interactions)
    }

def create_drug_pairs(dataset):
    """Create pairs of drugs with their interaction labels."""
    n_drugs = len(dataset['chemical_features'])
    pairs = []
    labels = []
    
    for i in range(n_drugs):
        for j in range(i+1, n_drugs):
            # Drug 1 data
            drug1 = {
                'chemical': dataset['chemical_features'][i],
                'protein': dataset['protein_features'][i],
                'protein_edge_index': dataset['protein_edge_indices'][i],
                'enzyme': dataset['enzyme_features'][i],
                'pathway': dataset['pathway_features'][i]
            }
            
            # Drug 2 data
            drug2 = {
                'chemical': dataset['chemical_features'][j],
                'protein': dataset['protein_features'][j],
                'protein_edge_index': dataset['protein_edge_indices'][j],
                'enzyme': dataset['enzyme_features'][j],
                'pathway': dataset['pathway_features'][j]
            }
            
            pairs.append((drug1, drug2))
            labels.append(dataset['interactions'][i,j])
    
    return pairs, torch.FloatTensor(labels)

if __name__ == "__main__":
    # Generate dummy dataset
    dataset = generate_dummy_dataset(n_drugs=50)
    pairs, labels = create_drug_pairs(dataset)
    print(f"Generated {len(pairs)} drug pairs with {labels.sum().item()} positive interactions") 