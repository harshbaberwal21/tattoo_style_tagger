"""Module containing the training and evaluation function."""

import torch
from .dataset_loader import TattooDataset, get_data_loader, get_image_tranforms
from torcheval.metrics import MultilabelAccuracy
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

    batch_size = 64
    learning_rate = 0.03
    num_epochs = 3
    device = get_device()


    label_index_map = get_label_index_map(tattoos_meta_data)
    out_labels_count = len(label_index_map)
    image_transforms = get_image_tranforms()
    
# TODO: Refactor to remove code redundancy 
    train_tattoo_ids = get_tattoo_ids(tattoos_meta_data, 'train')
    train_tattoo_labels = get_labels(tattoos_meta_data, 'train')
    train_dataset = TattooDataset(train_tattoo_ids, train_tattoo_labels, label_index_map, IMAGE_DIR, transform=image_transforms)
    train_tattoo_data_loader = get_data_loader(train_dataset, batch_size)

    test_tattoo_ids = get_tattoo_ids(tattoos_meta_data, 'test')
    test_tattoo_labels = get_labels(tattoos_meta_data, 'test')
    test_dataset = TattooDataset(test_tattoo_ids, test_tattoo_labels, label_index_map, IMAGE_DIR, transform=image_transforms)
    test_tattoo_data_loader = get_data_loader(test_dataset, batch_size)

    # Define Model
    model = get_model(out_labels_count).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_cached_activations = cache_frozen_network_activations(model.frozen, train_tattoo_data_loader, device)
    testing_cached_activations = cache_frozen_network_activations(model.frozen, test_tattoo_data_loader, device)

    for epoch in range(1, num_epochs+1):
        avg_training_loss = training_loop(model.unfrozen, training_cached_activations, train_tattoo_data_loader, optimizer, loss_function, device)
        avg_testing_loss, exact_match_accuracy, hamming_accuracy = \
            testing_loop(model.unfrozen, testing_cached_activations, test_tattoo_data_loader, loss_function, device)

        print(f"\tTraining Epoch: {epoch}\n",
        f"Average Training Loss: {avg_training_loss}, ",
        f"Average Testing Loss: {avg_testing_loss}, ",
        f"Testing Exact Accuracy: {exact_match_accuracy}, ",
        f"Testing Hamming Accuracy: {hamming_accuracy}")

    # save_model(model, MODEL_DIR+'model.pt')

    return model, optimizer, loss_function


def training_loop(unfrozen, training_cached_activations, train_data_loader, optimizer, loss_function, device):

    num_batches = len(train_data_loader)
    total_training_loss = 0
    unfrozen.train()
    for batch, (_, train_labels) in enumerate(train_data_loader):
        # Compute prediction and loss
        frozen_layer_activations = training_cached_activations[batch]
        train_labels = train_labels.to(device)
        predicted_logits = unfrozen(frozen_layer_activations)
        loss = loss_function(predicted_logits, train_labels)
        total_training_loss += loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_training_loss = total_training_loss/num_batches
    return avg_training_loss


def testing_loop(unfrozen, testing_cached_activations, test_data_loader, loss_function, device):

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

    exact_match_accuracy
    hamming_accuracy.compute()
    avg_testing_loss = total_testing_loss/num_batches

    return avg_testing_loss.item(), exact_match_accuracy.compute().item(), hamming_accuracy.compute().item()


def cache_frozen_network_activations(frozen_layers, tattoo_data_loader, device):
    frozen_layers.eval()
    cached_activations = {}
    with torch.no_grad():
        for batch, (train_features, _) in enumerate(tattoo_data_loader):
            activations = frozen_layers(train_features.to(device))
            cached_activations[batch] = torch.flatten(activations,1)
    return cached_activations
