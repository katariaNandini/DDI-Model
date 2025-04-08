import torch
import torch.nn as nn
import torch.nn.functional as F

class ChemicalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ChemicalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

class ProteinEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ProteinEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index=None):  # edge_index is kept for compatibility but not used
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x.mean(dim=0)  # Global mean pooling

class EnzymeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EnzymeEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

class PathwayEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PathwayEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x):
        # Add batch dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        output, (hidden, _) = self.lstm(x)
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        return self.fc(hidden.squeeze(0))

class DDIMDL(nn.Module):
    def __init__(self, chemical_dim, protein_dim, enzyme_dim, pathway_dim, hidden_dim):
        super(DDIMDL, self).__init__()
        
        # Feature encoders
        self.chemical_encoder = ChemicalEncoder(chemical_dim, hidden_dim)
        self.protein_encoder = ProteinEncoder(protein_dim, hidden_dim)
        self.enzyme_encoder = EnzymeEncoder(enzyme_dim, hidden_dim)
        self.pathway_encoder = PathwayEncoder(pathway_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, drug1_data, drug2_data):
        # Encode drug1 features
        chem1 = self.chemical_encoder(drug1_data['chemical'])
        prot1 = self.protein_encoder(drug1_data['protein'], drug1_data['protein_edge_index'])
        enzy1 = self.enzyme_encoder(drug1_data['enzyme'])
        path1 = self.pathway_encoder(drug1_data['pathway'])
        
        # Encode drug2 features
        chem2 = self.chemical_encoder(drug2_data['chemical'])
        prot2 = self.protein_encoder(drug2_data['protein'], drug2_data['protein_edge_index'])
        enzy2 = self.enzyme_encoder(drug2_data['enzyme'])
        path2 = self.pathway_encoder(drug2_data['pathway'])
        
        # Concatenate all features
        combined = torch.cat([chem1, prot1, enzy1, path1, chem2, prot2, enzy2, path2], dim=0)
        
        # Predict interaction
        return self.fusion(combined) 