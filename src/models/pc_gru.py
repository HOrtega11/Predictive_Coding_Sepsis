import torch
import torch.nn as nn


class PredictiveCodingGRU(nn.Module):
    """
    Predictive-coding-inspired GRU for multivariate time-series forecasting.

    This model extends a standard GRU by explicitly computing prediction errors
    inside the forward pass. At each time step and layer, the model:

    1. Uses the current recurrent state to predict the current layer input.
    2. Computes a prediction error.
    3. Propagates that error upward to the next layer.
    4. Updates recurrent states using the current input, prediction error,
       and top-down information from the higher layer when available.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.

    hidden_dim : int
        Hidden-state dimension for each GRU layer.

    output_dim : int
        Number of output features predicted by the model.

    num_layers : int, optional
        Number of predictive-coding recurrent layers. Default is 2.

    dropout : float, optional
        Dropout probability applied before the final forecast layer. Default is 0.1.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Layer input dimensions:
        # layer 0 receives the raw input features.
        # higher layers receive transformed prediction errors.
        self.layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1)

        self.gru_cells = nn.ModuleList()
        self.prediction_layers = nn.ModuleList()
        self.error_to_next = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            current_dim = self.layer_dims[layer_idx]

            # All layers receive A_l and E_l.
            # Lower layers also receive top-down input from the layer above.
            top_down_dim = hidden_dim if layer_idx < num_layers - 1 else 0
            gru_input_dim = current_dim + current_dim + top_down_dim

            self.gru_cells.append(
                nn.GRUCell(
                    input_size=gru_input_dim,
                    hidden_size=hidden_dim,
                )
            )

            # Predict the current layer representation from its recurrent state.
            self.prediction_layers.append(
                nn.Linear(hidden_dim, current_dim)
            )

            if layer_idx < num_layers - 1:
                # Transform prediction error into the representation used by
                # the next layer.
                self.error_to_next.append(
                    nn.Linear(current_dim, self.layer_dims[layer_idx + 1])
                )

                # Transform higher-layer recurrent state into top-down input.
                self.top_down_layers.append(
                    nn.Linear(hidden_dim, hidden_dim)
                )

        self.dropout = nn.Dropout(dropout)
        self.forecast_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Run a forward pass through the predictive-coding GRU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape:

            (batch_size, sequence_length, input_dim)

        Returns
        -------
        torch.Tensor
            Forecast tensor with shape:

            (batch_size, output_dim)

        Notes
        -----
        The model predicts one future time step from the full input window.
        The prediction horizon itself is handled by the dataset/windowing code,
        not by this model directly.
        """

        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize recurrent states for all layers at t = 0.
        hidden_states = [
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

        for t in range(seq_len):
            # ------------------------------------------------------------
            # Phase 1: Bottom-up prediction error computation
            # ------------------------------------------------------------
            #
            # layer_inputs[0] is the raw input at the current time step.
            # Higher layer inputs are generated from lower-layer errors.
            layer_inputs = [x[:, t, :]]
            errors = []

            for layer_idx in range(self.num_layers):
                A_l = layer_inputs[layer_idx]

                # Predict current layer input from previous hidden state.
                Ahat_l = self.prediction_layers[layer_idx](
                    hidden_states[layer_idx]
                )

                # Signed prediction error.
                E_l = A_l - Ahat_l
                errors.append(E_l)

                # Pass transformed error upward to the next layer.
                if layer_idx < self.num_layers - 1:
                    A_next = self.error_to_next[layer_idx](E_l)
                    layer_inputs.append(A_next)

            # ------------------------------------------------------------
            # Phase 2: Recurrent state update
            # ------------------------------------------------------------
            #
            # Update from top layer to bottom layer so that lower layers
            # receive updated same-time-step top-down information.
            new_hidden_states = [None] * self.num_layers

            for layer_idx in reversed(range(self.num_layers)):
                A_l = layer_inputs[layer_idx]
                E_l = errors[layer_idx]

                if layer_idx < self.num_layers - 1:
                    # Use updated higher-layer state R_{l+1}^t as top-down input.
                    top_down = self.top_down_layers[layer_idx](
                        new_hidden_states[layer_idx + 1]
                    )

                    gru_input = torch.cat([A_l, E_l, top_down], dim=1)

                else:
                    # Top layer has no higher layer, so no top-down input.
                    gru_input = torch.cat([A_l, E_l], dim=1)

                new_hidden_states[layer_idx] = self.gru_cells[layer_idx](
                    gru_input,
                    hidden_states[layer_idx],
                )

            hidden_states = new_hidden_states

        # Use the final bottom-layer recurrent state for forecasting.
        prediction = self.forecast_layer(
            self.dropout(hidden_states[0])
        )

        return prediction
    