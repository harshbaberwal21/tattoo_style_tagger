"""Module containing the training and evaluation function."""

# pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments

from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics import MultilabelAccuracy
from .dataset_loader import get_precomputed_data_loader
from .model_utils import get_device
from .model_definition import get_model


from .model_utils import (
    get_label_index_map,
    MODEL_DIR,
    get_predicted_labels,
    save_model,
)


def train_model(tattoos_meta_data, num_epochs = 10, batch_size = 64, pretrained_model_path: str = None):
    """Train the model.

    Args:
        tattoos_meta_data (_type_): Tattoos metadata, processed, augmented
        and with example_type indicator
    """

    device = get_device()
    label_index_map = get_label_index_map(tattoos_meta_data)
    out_labels_count = len(label_index_map)

    # Define Model, loss function and optimizer
    model = get_model(out_labels_count, pretrained_model_path).to(device)

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    learning_rate_scheduler = ReduceLROnPlateau(
        optimizer = optimizer,
        mode = 'min',
        factor = 1/3,
        min_lr = 0.003
        )

    # Define precomputed data loaders
    train_precomputed_tattoo_data_loader = get_precomputed_data_loader(
        model,
        tattoos_meta_data,
        label_index_map,
        batch_size,
        device,
        example_type="train",
    )
    val_precomputed_tattoo_data_loader = get_precomputed_data_loader(
        model,
        tattoos_meta_data,
        label_index_map,
        batch_size,
        device,
        example_type="val",
    )
    test_precomputed_tattoo_data_loader = get_precomputed_data_loader(
        model,
        tattoos_meta_data,
        label_index_map,
        batch_size,
        device,
        example_type="test",
    )

    eval_metrics = pd.DataFrame(
        columns=[
            "avg_training_loss",
            "training_exact_match_accuracy",
            "training_hamming_accuracy",
            "avg_val_loss",
            "val_exact_match_accuracy",
            "val_hamming_accuracy",
        ]
    )

    losses_during_training = {}
    for epoch in range(1, num_epochs + 1):
        training_loss_by_minibatch = training_loop(
            model.unfrozen,
            train_precomputed_tattoo_data_loader,
            optimizer,
            loss_function,
            device,
        )

        losses_during_training[epoch] = training_loss_by_minibatch
        pd.DataFrame(
            data={
                "batch": training_loss_by_minibatch.keys(),
                "loss": training_loss_by_minibatch.values(),
            }
        ).plot(x="batch", y="loss", kind="line")
        plt.show()

        avg_training_loss, training_exact_match_accuracy, training_hamming_accuracy = (
            get_evalulation_metrics(
                model.unfrozen,
                train_precomputed_tattoo_data_loader,
                loss_function,
                device,
            )
        )

        avg_val_loss, val_exact_match_accuracy, val_hamming_accuracy = (
            get_evalulation_metrics(
                model.unfrozen,
                val_precomputed_tattoo_data_loader,
                loss_function,
                device,
            )
        )

        eval_metrics.loc[epoch - 1] = {
            "epoch": epoch,
            "avg_training_loss": avg_training_loss,
            "training_exact_match_accuracy": training_exact_match_accuracy,
            "training_hamming_accuracy": training_hamming_accuracy,
            "avg_val_loss": avg_val_loss,
            "val_exact_match_accuracy": val_exact_match_accuracy,
            "val_hamming_accuracy": val_hamming_accuracy,
        }

        print(eval_metrics.tail(1))

        learning_rate_scheduler.step(avg_val_loss)

    avg_testing_loss, testing_exact_match_accuracy, testing_hamming_accuracy = (
        get_evalulation_metrics(
            model.unfrozen,
            test_precomputed_tattoo_data_loader,
            loss_function,
            device,
        )
    )

    test_metrics = {
        "avg_testing_loss": avg_testing_loss,
        "testing_exact_match_accuracy": testing_exact_match_accuracy,
        "testing_hamming_accuracy": testing_hamming_accuracy,
    }

    save_model(model, MODEL_DIR + "model.bin")

    return (
        model,
        optimizer,
        loss_function,
        losses_during_training,
        eval_metrics,
        test_metrics,
    )


def training_loop(
    unfrozen,
    train_data_loader,
    optimizer,
    loss_function,
    device,
):
    """Train the model for one epoch.

    Args:
        unfrozen (pytorch.nn.module): the model portion to be trained.
        Should have requires_grad set to True.
        train_data_loader (torch.utils.data.DataLoader): training DataLoader object
        optimizer (torch.optim.adam.Adam): optimizer to use
        loss_function (torch.nn.modules.loss.BCEWithLogitsLoss): loss function to use
        device (str): The device available for processing (cpu, gpu or mps).

    Returns:
        _type_: _description_
    """
    loss_by_minibatch = {}
    unfrozen.train()
    for batch, (train_acts, train_labels) in enumerate(train_data_loader):
        # Forward-Propagation
        train_labels = train_labels.to(device)
        predicted_logits = unfrozen(train_acts)

        # Compute loss
        loss = loss_function(predicted_logits, train_labels)
        loss_by_minibatch[batch + 1] = loss.item()

        # Back-Propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss_by_minibatch


def get_evalulation_metrics(unfrozen, data_loader, loss_function, device):
    """Test and evaluate the model through all batches of one epoch.

    Args:
        unfrozen (pytorch.nn.module): the model portion to be tested and
        that results the label predictions.
        testing_cached_activations (dict of torch.Tensor): cached activations of the testing
        dataset passed through the frozen layers.
        test_data_loader (torch.utils.data.DataLoader): testing DataLoader object
        loss_function (torch.nn.modules.loss.BCEWithLogitsLoss): loss function to use
        device (str): The device available for processing (cpu, gpu or mps).

    Returns:
        _type_: _description_
    """
    size = len(data_loader.dataset)
    total_loss = 0
    exact_match_accuracy = MultilabelAccuracy()
    hamming_accuracy = MultilabelAccuracy(criteria="hamming")

    unfrozen.eval()
    with torch.no_grad():
        for _, (acts, labels) in enumerate(data_loader):
            labels = labels.to(device)
            predicted_logits = unfrozen(acts)
            total_loss += loss_function(predicted_logits, labels)

            predicted_labels_indices = get_predicted_labels(predicted_logits)
            exact_match_accuracy.update(predicted_labels_indices, labels)
            hamming_accuracy.update(predicted_labels_indices, labels)

    avg_loss = total_loss / size

    return (
        avg_loss.item(),
        exact_match_accuracy.compute().item(),
        hamming_accuracy.compute().item(),
    )
