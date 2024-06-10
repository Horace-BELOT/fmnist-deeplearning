"""Training of Fashion MNIST dataset"""
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from typing import Optional
import matplotlib.pyplot as plt

from data import DatasetDownLoader, DatasetTransformer
from model import LinearNet, FullyConnected, FullyConnectedRegularized, VanillaCNN
from utils import train, test, ModelCheckpoint, compute_mean_std

from enums import Enum

class MODEL_TYPE(Enum):
    LINEAR_NET = "LINEAR_NET"
    FULLY_CONNECTED = "FULLY_CONNECTED"
    FULLY_CONNECTED_REGULARIZED = "FULLY_CONNECTED_REGULARIZED"
    VANILLA_CNN = "VANILLA_CNN"

class DATASET(Enum):
    MNIST = "MNIST"
    FASHION_MNIST = "FASHION_MNIST"


def main(load_model_path: Optional[str] = None,
         dataset_src: DATASET = DATASET.MNIST,
         model_type: MODEL_TYPE = MODEL_TYPE.LINEAR_NET) -> None:
    """
    Training and Testing model on MNIST Fashion or MNIST dataset

    Args:
        load_model_path: str | None
            if desired, path of the pretrained model to load
    """
    silent: bool = False
    num_threads = 4     # Loading the dataset is using 4 CPU threads
    batch_size  = 128   # Using minibatches of 128 samples
    
    NORMALIZING: bool = True

    # Load data
    if dataset_src == DATASET.MNIST:
        print("Loading dataset: MNIST")
        train_ds, valid_ds, test_ds = DatasetDownLoader.mnist(validation_ratio=0.2)    
    elif dataset_src == DATASET.FASHION_MNIST:
        print("Loading dataset: FashionMNIST")
        train_ds, valid_ds, test_ds = DatasetDownLoader.fashion_mnist(validation_ratio=0.2)
    else:
        raise ValueError(f"{dataset_src} not implemented as dataset")

    # For now the transform function is a simple toTensor() function
    # (i.e it transforms numpy array or pil image to a pytorch tensor)
    f_transform = transforms.ToTensor()
    
    if NORMALIZING:
        #
        normalizing_dataset = DatasetTransformer(train_ds, transforms.ToTensor())
        normalizing_loader = DataLoader(dataset=normalizing_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads)
        # Compute mean and variance from the training set
        mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)
        # Adding this to the transform
        f_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - mean_train_tensor)/std_train_tensor)
        ])

    # The transform is then applied to all datasets
    train_ds = DatasetTransformer(train_ds, f_transform)
    valid_ds = DatasetTransformer(valid_ds, f_transform)
    test_ds = DatasetTransformer(test_ds, f_transform)

    # Creating dataloaders
    kwargs = {"batch_size": batch_size, "shuffle": True, "num_workers": num_threads}
    train_loader = DataLoader(dataset=train_ds, **kwargs)
    valid_loader = DataLoader(dataset=valid_ds, **kwargs)
    test_loader = DataLoader(dataset=test_ds, **kwargs)

    if not silent:
        print(f"The train set contains {len(train_loader.dataset)} images, in {len(train_loader)} batches")
        print(f"The validation set contains {len(valid_loader.dataset)} images, in {len(valid_loader)} batches")
        print(f"The test set contains {len(test_loader.dataset)} images, in {len(test_loader)} batches")

    # Create model
    model: torch.nn.Module

    if model_type == MODEL_TYPE.LINEAR_NET:
        model = LinearNet(n_input=1 * 28 * 28, n_classes=10)
    elif model_type == MODEL_TYPE.FULLY_CONNECTED:
        model = FullyConnected(n_input=1 * 28 * 28, n_classes=10)
    elif model_type == MODEL_TYPE.FULLY_CONNECTED_REGULARIZED:
        model = FullyConnectedRegularized(n_input=1 * 28 * 28, n_classes=10, l2_reg=1e-2)
    elif model_type == MODEL_TYPE.VANILLA_CNN:
        model = VanillaCNN(num_classes=10)

    if load_model_path is not None:
        # If path is provided, load the model
        model.load_state_dict(torch.load(load_model_path))
    
    # Whether or not to use GPU (if cuda is available)
    use_gpu: bool = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Passing the model on the device
    model.to(device)

    # Initializing the loss
    f_loss = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters())

    # Model Checkpoint (to save best model)
    model_checkpoint = ModelCheckpoint(
        model, log_name=f"{str(model_type)}_FASHION_MNIST_%", silent=False, tensorboard_logdir="logs/tensorboard")

    epochs: int = 100
    for t in range(epochs):
        print("Epoch {}".format(t))
        train_loss, train_acc = train(model, train_loader, f_loss, optimizer, device)
        print("    Training :   Loss : {:.4f}, Acc : {:.4f}".format(train_loss, train_acc))
        val_loss, val_acc = test(model, valid_loader, f_loss, device)
        print("    Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))
        model_checkpoint.update(val_loss)
        model_checkpoint.log_results({"train_accuracy": train_acc,
                                      "train_loss": train_loss,
                                      "valid_accuracy": val_acc,
                                      "valid_loss": val_loss})
        
        # Plotting some results
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        x_arr = model_checkpoint.df.index
        # Accuracy subplot
        ax1.plot(x_arr, model_checkpoint.df["valid_accuracy"], label="valid")
        ax1.plot(x_arr, model_checkpoint.df["train_accuracy"], label="train")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        # Loss subplot
        ax2.plot(x_arr, model_checkpoint.df["valid_loss"], label="valid")
        ax2.plot(x_arr, model_checkpoint.df["train_loss"], label="train")
        ax2.set_ylabel("Loss")
        ax2.legend()
        plt.xlabel("Epoch")
        plt.savefig(os.path.join(model_checkpoint.log_path, "loss_and_acc.png"))


if __name__ == "__main__":
    main(
        # load_model_path=None,
        dataset_src=DATASET.MNIST,
        model_type=MODEL_TYPE.VANILLA_CNN,
    )
    
    