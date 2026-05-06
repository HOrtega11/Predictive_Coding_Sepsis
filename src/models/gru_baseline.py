import torch.nn as nn


class GRUBaseline(nn.Module):
    """
    Standard GRU baseline model for multivariate time-series forecasting.

    This model processes an input sequence using a single-layer GRU and uses
    the final hidden representation (last time step) to predict the target
    vital signs at a future time point.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step (e.g., number of vital signs).

    hidden_dim : int
        Number of hidden units in the GRU.

    output_dim : int
        Number of output features (typically equal to input_dim for forecasting).

    dropout : float, optional
        Dropout probability applied before the final linear layer. Default is 0.1.

    Notes
    -----
    - This is a single-layer GRU (no stacking).
    - The model uses the final time step representation for prediction.
    - This serves as the primary deep learning baseline for comparison
      with the predictive coding GRU (PC-GRU).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        # GRU layer processes the input sequence
        # Input shape: (batch_size, seq_len, input_dim)
        # Output shape: (batch_size, seq_len, hidden_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Dropout applied before the output layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Linear layer maps final hidden state to prediction
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the GRU baseline model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns
        -------
        torch.Tensor
            Predicted output of shape (batch_size, output_dim)

        Process
        -------
        1. Pass sequence through GRU
        2. Extract hidden representation from final time step
        3. Apply dropout for regularization
        4. Map to output space using linear layer
        """

        # Pass input sequence through GRU
        gru_out, _ = self.gru(x)

        # Extract the output corresponding to the final time step
        # Shape: (batch_size, hidden_dim)
        last_hidden = gru_out[:, -1, :]

        # Apply dropout and project to output dimension
        prediction = self.output_layer(self.dropout(last_hidden))

        return prediction