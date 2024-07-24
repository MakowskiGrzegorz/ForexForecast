import torch
from torch import nn

__DEVICE__ = 'cuda' if torch.cuda.is_available() else 'cpu'
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True).to(__DEVICE__)
        self.fc = nn.Linear(hidden_dim, output_dim).to(__DEVICE__)
        self.act = nn.Sigmoid()
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(__DEVICE__)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(__DEVICE__)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        out = self.act(out)
        return out

