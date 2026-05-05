
import torch


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        predictions = model(x)
        loss = loss_fn(predictions, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_last_inputs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            predictions = model(x)
            loss = loss_fn(predictions, y)

            total_loss += loss.item()

            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())

            # Last observed timestep (used for direction accuracy)
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
