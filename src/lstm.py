import torch
from torch import nn

__DEVICE__ = 'cuda' if torch.cuda.is_available() else 'cpu'
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,dropout=dropout).to(__DEVICE__)
        self.fc = nn.Linear(hidden_dim, output_dim).to(__DEVICE__)
        self.act = nn.Tanh()
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(__DEVICE__)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(__DEVICE__)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        out = self.act(out)
        return out

class LSTMModelSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout, future_steps):
        super(LSTMModelSeq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.future_steps = future_steps
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout).to(__DEVICE__)
        self.fc = nn.Linear(hidden_dim, output_dim).to(__DEVICE__)
        self.act = nn.Tanh()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(__DEVICE__)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(__DEVICE__)
        
        outputs = []
        for step in range(self.future_steps):
            # LSTM forward pass
            out, (h0, c0) = self.lstm(x, (h0.detach(), c0.detach()))
            # Fully connected layer to get the prediction
            pred_price = self.fc(out[:, -1, :])
            pred_price = self.act(pred_price)

            # Save the prediction
            outputs.append(pred_price)
            
            # Create the new input for the next step
            # Assuming the target feature (price) is the first feature in the input
            new_input = x.clone()
            new_input = torch.cat((x[:, 1:, :], torch.cat([pred_price, x[:, -1, 1:]], dim=1).unsqueeze(1)), dim=1)
            
            # Update x for the next iteration
            x = new_input

        # Stack all the predicted future steps
        return torch.cat(outputs, dim=1).squeeze(-1)
    

class LSTMModelMultiStep(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout, future_steps):
        super(LSTMModelMultiStep, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.future_steps = future_steps
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout).to(__DEVICE__)
        self.fc = nn.Linear(hidden_dim, output_dim * future_steps).to(__DEVICE__)
        self.act = nn.Tanh()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(__DEVICE__)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(__DEVICE__)
        
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        out = self.act(out)
        # Reshape output to (batch_size, future_steps, output_dim)
        return out.view(-1, out.size(-1))