
import torch.nn as nn


class GRUBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        prediction = self.output_layer(self.dropout(last_hidden))
        return prediction
    