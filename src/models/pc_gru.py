
import torch
import torch.nn as nn


class PredictiveCodingGRU(nn.Module):
    """
    Predictive-coding-inspired GRU model.

    At each time step and layer:
    A_l      = input/target representation at layer l
    R_l      = recurrent representation at layer l
    Ahat_l   = prediction generated from R_l
    E_l      = prediction error: A_l - Ahat_l
    A_{l+1}  = transformed error passed upward
    R_l      = updated using A_l, E_l, and, when available, top-down R_{l+1}

    Unlike a standard GRU, prediction error is explicitly computed and fed
    into the recurrent update during the forward pass.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1)

        self.gru_cells = nn.ModuleList()
        self.prediction_layers = nn.ModuleList()
        self.error_to_next = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()

        for l in range(num_layers):
            current_dim = self.layer_dims[l]
            top_down_dim = hidden_dim if l < num_layers - 1 else 0

            gru_input_dim = current_dim + current_dim + top_down_dim

            self.gru_cells.append(
                nn.GRUCell(
                    input_size=gru_input_dim,
                    hidden_size=hidden_dim,
                )
            )

            self.prediction_layers.append(
                nn.Linear(hidden_dim, current_dim)
            )

            if l < num_layers - 1:
                self.error_to_next.append(
                    nn.Linear(current_dim, self.layer_dims[l + 1])
                )

                self.top_down_layers.append(
                    nn.Linear(hidden_dim, hidden_dim)
                )

        self.dropout = nn.Dropout(dropout)
        self.forecast_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device

        hidden_states = [
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

        for t in range(seq_len):
            layer_inputs = [x[:, t, :]]
            errors = []

            for l in range(self.num_layers):
                A_l = layer_inputs[l]

                Ahat_l = self.prediction_layers[l](hidden_states[l])
                E_l = A_l - Ahat_l

                errors.append(E_l)

                if l < self.num_layers - 1:
                    A_next = self.error_to_next[l](E_l)
                    layer_inputs.append(A_next)

            new_hidden_states = []

            for l in range(self.num_layers):
                A_l = layer_inputs[l]
                E_l = errors[l]

                if l < self.num_layers - 1:
                    top_down = self.top_down_layers[l](hidden_states[l + 1])
                    gru_input = torch.cat([A_l, E_l, top_down], dim=1)
                else:
                    gru_input = torch.cat([A_l, E_l], dim=1)

                R_l = self.gru_cells[l](gru_input, hidden_states[l])
                new_hidden_states.append(R_l)

            hidden_states = new_hidden_states

        prediction = self.forecast_layer(self.dropout(hidden_states[0]))

        return prediction
    
