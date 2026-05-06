import torch


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Train a model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model being trained.

    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of input windows and targets.

    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.

    loss_fn : callable
        Loss function, such as torch.nn.MSELoss.

    device : torch.device
        Device used for training, either CPU or CUDA.

    Returns
    -------
    float
        Average training loss across all batches.
    """

    # Set model to training mode so dropout and other training-specific
    # operations are active.
    model.train()

    total_loss = 0

    for x, y in dataloader:
        # Move batch to CPU or GPU.
        x = x.to(device)
        y = y.to(device)

        # Clear gradients from the previous optimization step.
        optimizer.zero_grad()

        # Forward pass: predict future vital signs.
        predictions = model(x)

        # Compute prediction loss.
        loss = loss_fn(predictions, y)

        # Backpropagate gradients and update model weights.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Avoid division by zero if dataloader is empty.
    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate a model on a validation or test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Model being evaluated.

    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of input windows and targets.

    loss_fn : callable
        Loss function used to compute evaluation loss.

    device : torch.device
        Device used for evaluation, either CPU or CUDA.

    Returns
    -------
    tuple
        avg_loss : float
            Average loss across all evaluation batches.

        all_predictions : torch.Tensor
            Model predictions with shape (num_examples, output_dim).

        all_targets : torch.Tensor
            True target values with shape (num_examples, output_dim).

        all_last_inputs : torch.Tensor
            Last observed input timestep from each window with shape
            (num_examples, input_dim). This is used for directional accuracy.

    Notes
    -----
    Gradient computation is disabled during evaluation to reduce memory usage
    and prevent model updates.
    """

    # Set model to evaluation mode so dropout is disabled.
    model.eval()

    total_loss = 0
    all_predictions = []
    all_targets = []
    all_last_inputs = []

    with torch.no_grad():
        for x, y in dataloader:
            # Move batch to CPU or GPU.
            x = x.to(device)
            y = y.to(device)

            # Forward pass only; no gradient computation.
            predictions = model(x)

            # Compute validation/test loss.
            loss = loss_fn(predictions, y)
            total_loss += loss.item()

            # Move tensors back to CPU for metric calculation.
            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())

            # Store the last observed timestep in each input window.
            # This is used to determine whether the model predicted the
            # correct direction of change.
            all_last_inputs.append(x[:, -1, :].cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_last_inputs = torch.cat(all_last_inputs)

    return (
        total_loss / max(len(dataloader), 1),
        all_predictions,
        all_targets,
        all_last_inputs,
    )