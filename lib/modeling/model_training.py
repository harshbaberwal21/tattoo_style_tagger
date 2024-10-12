"""Module containing the training and evaluation function."""

# pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments

import torch
from torcheval.metrics import MultilabelAccuracy
from .dataset_loader import TattooDataset, get_data_loader, get_image_tranforms
from ..utils import get_device
from .model_definition import get_model


from .model_utils import (
    get_label_index_map,
    get_tattoo_ids,
    get_labels,
    IMAGE_DIR,
    # MODEL_DIR,
    get_predicted_labels,
    # save_model,
)


def train_model(tattoos_meta_data):
    """Train the model.

    Args:
        tattoos_meta_data (_type_): Tattoos metadata, processed, augmented
        and with example_type indicator
    """
    batch_size = 64
    learning_rate = 0.03
    num_epochs = 3
    device = get_device()

    label_index_map = get_label_index_map(tattoos_meta_data)
    out_labels_count = len(label_index_map)
    image_transforms = get_image_tranforms()

    train_tattoo_ids = get_tattoo_ids(tattoos_meta_data, "train")
    train_tattoo_labels = get_labels(tattoos_meta_data, "train")
    train_dataset = TattooDataset(
        train_tattoo_ids,
        train_tattoo_labels,
        label_index_map,
        IMAGE_DIR,
        transform=image_transforms,
    )
    train_tattoo_data_loader = get_data_loader(train_dataset, batch_size)

    test_tattoo_ids = get_tattoo_ids(tattoos_meta_data, "test")
    test_tattoo_labels = get_labels(tattoos_meta_data, "test")
    test_dataset = TattooDataset(
        test_tattoo_ids,
        test_tattoo_labels,
        label_index_map,
        IMAGE_DIR,
        transform=image_transforms,
    )
    test_tattoo_data_loader = get_data_loader(test_dataset, batch_size)

    # Define Model
    model = get_model(out_labels_count).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_cached_activations = cache_frozen_network_activations(
        model.frozen, train_tattoo_data_loader, device
    )
    testing_cached_activations = cache_frozen_network_activations(
        model.frozen, test_tattoo_data_loader, device
    )

    for epoch in range(1, num_epochs + 1):
        training_loop(
            model.unfrozen,
            training_cached_activations,
            train_tattoo_data_loader,
            optimizer,
            loss_function,
            device,
        )
        avg_testing_loss, exact_match_accuracy, hamming_accuracy = testing_loop(
            model.unfrozen,
            testing_cached_activations,
            test_tattoo_data_loader,
            loss_function,
            device,
        )

        print(
            f"\tTraining Epoch: {epoch}\n",
            # f"Average Training Loss: {avg_training_loss}\n",
            f"Average Testing Loss: {avg_testing_loss}\n",
            f"Testing Exact Accuracy: {exact_match_accuracy}\n",
            f"Testing Hamming Accuracy: {hamming_accuracy}",
        )

    # save_model(model, MODEL_DIR+'model.pt')

    return model, optimizer, loss_function


def training_loop(
    unfrozen,
    training_cached_activations,
    train_data_loader,
    optimizer,
    loss_function,
    device,
):
    """Train the model for one epoch.

    Args:
        unfrozen (pytorch.nn.module): the model portion to be trained.
        Should have requires_grad set to True.
        training_cached_activations (dict of torch.Tensor): cached activations of the training
        dataset passed through the frozen layers.
        train_data_loader (torch.utils.data.DataLoader): training DataLoader object
        optimizer (torch.optim.adam.Adam): optimizer to use
        loss_function (torch.nn.modules.loss.BCEWithLogitsLoss): loss function to use
        device (str): The device available for processing (cpu, gpu or mps).

    Returns:
        _type_: _description_
    """
    unfrozen.train()
    for batch, (_, train_labels) in enumerate(train_data_loader):
        # Compute prediction and loss
        frozen_layer_activations = training_cached_activations[batch]
        train_labels = train_labels.to(device)
        predicted_logits = unfrozen(frozen_layer_activations)
        loss = loss_function(predicted_logits, train_labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def testing_loop(
    unfrozen, testing_cached_activations, test_data_loader, loss_function, device
):
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
    # size = len(test_data_loader.dataset)
    num_batches = len(test_data_loader)

    total_testing_loss = 0
    exact_match_accuracy = MultilabelAccuracy()
    hamming_accuracy = MultilabelAccuracy(criteria="hamming")

    unfrozen.eval()
    with torch.no_grad():
        for batch, (_, test_labels) in enumerate(test_data_loader):
            frozen_layer_activations = testing_cached_activations[batch]
            test_labels = test_labels.to(device)
            predicted_logits = unfrozen(frozen_layer_activations)
            total_testing_loss += loss_function(predicted_logits, test_labels)

            predicted_labels_indices = get_predicted_labels(predicted_logits)
            exact_match_accuracy.update(predicted_labels_indices, test_labels)
            hamming_accuracy.update(predicted_labels_indices, test_labels)

    avg_testing_loss = total_testing_loss / num_batches

    return (
        avg_testing_loss.item(),
        exact_match_accuracy.compute().item(),
        hamming_accuracy.compute().item(),
    )


def cache_frozen_network_activations(frozen_layers, tattoo_data_loader, device):
    """Pass data through fixed/frozen network and cache activations for faster training.

    Args:
        frozen_layers (pytorch.nn.module): the model portion that is frozen i.e.
        has the requires_grad attribute of the parameters set to False.
        tattoo_data_loader (torch.utils.data.DataLoader): DataLoader object for
        the data to pass through
        device (str): The device available for processing (cpu, gpu or mps).

    Returns:
       dict of torch.Tensor: cached activations
    """
    frozen_layers.eval()
    cached_activations = {}
    with torch.no_grad():
        for batch, (train_features, _) in enumerate(tattoo_data_loader):
            activations = frozen_layers(train_features.to(device))
            cached_activations[batch] = torch.flatten(activations, 1)
    return cached_activations
