

import numpy as np
import math
import torch

def train_model(model, criterion, optimizer, train_loader, val_loader, scaler,
                tracker, args, device):

    # Performance metrics and tracking
    val_loss, top1, top5, top10, top25 = \
        validate(val_loader, model, criterion, scaler, device)
    tracker.track(0, 0, val_loss, top1, top5, top10, top25)

    for epoch in range(1, args.n_epochs + 1):

        for X_batch, labels_batch in train_loader:

            # Copy data to device
            X_batch, labels_batch = X_batch.to(device), labels_batch.to(device)

            # Scale X
            X_batch = scaler.normalize(X_batch)

            # Forward pass: Compute predicted y by passing x to the model
            out = model(X_batch)

            # Construct y tensor
            y_batch = torch.zeros_like(out) if model.mahalanobis else X_batch

            # Compute and print loss
            loss = criterion(out, y_batch)
            print('Epoch: {}/{} -- Loss: {}'.format(epoch, args.n_epochs,
                                                    loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model.mahalanobis:
                with torch.no_grad():
                    X_fit = model.reconstruct(X_batch)
                    model.mahalanobis_layer.update(X_batch, X_fit)

        # Performance metrics and tracking
        val_loss, top1, top5, top10, top25 = \
            validate(val_loader, model, criterion, scaler, device)
        tracker.track(epoch, loss, val_loss, top1, top5, top10, top25)

    return model, epoch

def outlier_factor(x, x_val):
    err = x - x_val
    err = torch.pow(err, 2)
    err = torch.sum(err, 1)
    return err / len(err)


def performance(anomalies, scores, percentage):

    # Order anomalies (binary vector) by the anomaly score in descending order
    _, ordering = torch.sort(scores, descending=True)
    ordered_anomalies = anomalies[ordering.type(torch.LongTensor)]

    # Number of observations to include in top
    n_top = math.ceil(len(anomalies) * percentage / 100)