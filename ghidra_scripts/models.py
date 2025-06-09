import torch.nn as nn

class ByteGRU(nn.Module):
    def __init__(self, emb_dim=16, hidden_dim=64, out_dim=32):
        super().__init__()
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        _, h_n = self.gru(x)
        return self.out(h_n.squeeze(0))
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=32, hidden_dim1=64, hidden_dim2=32):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x).squeeze(-1)
    
class ByteEmbedder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.Sigmoid()
        )

    def forward(self, byte_seq):
        if byte_seq.dim() == 1:
            byte_seq = byte_seq.unsqueeze(0)
        x = byte_seq.unsqueeze(1).float() / 255.0
        x = self.cnn(x).squeeze(-1)
        return self.proj(x)