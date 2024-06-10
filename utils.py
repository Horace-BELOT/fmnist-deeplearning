"""
Utility functions for running model training and testing in pytorch
"""
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from typing import Tuple, Optional, Dict, Any, List


def train(
        model: torch.nn.Module, 
        loader: DataLoader, 
        f_loss, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device) -> Tuple[float, float]:
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

                     
    Returns :
        - Training Loss
        - Training Accuracy
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    # Total number of inputs passed
    n_total: int = 0
    # Correct number of inputs
    n_correct: int = 0
    # Accumulated loss
    total_loss: float = 0

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        if model.regularized:  # If we have some regularization for our model, we regularize
            model.penalty().backward()
        optimizer.step()

        # Count
        n_total += inputs.shape[0]
        total_loss += inputs.shape[0] * f_loss(outputs, targets).item()
        n_correct += (outputs.argmax(dim=1) == targets).sum().item()
    return (total_loss / n_total), (n_correct / n_total) 


def test(model, loader: DataLoader, f_loss, device: torch.device) -> Tuple[float, float]:
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation 

    Returns :
        - Testing Loss
        - Testing Accuracy

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        for i, (inputs, targets) in enumerate(loader):

            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()
        return tot_loss/N, correct/N
    

class ModelCheckpoint:

    def __init__(
            self,
            model: torch.nn.Module,
            log_name: str = "linear_FMNIST_%",
            model_file_name: str = "best_model.pt",
            df_file_name: str = "data.csv",
            silent: bool = True,
            tensorboard_logdir: Optional[str] = None,
        ) -> None:
        """
        
        """
        self.min_loss: Optional[float] = None
        self.log_path: str = __class__.__generate_log_dir(name_template=log_name)
        self.model: torch.nn.Module = model
        self.silent: bool = silent
        self.df: pd.DataFrame = pd.DataFrame([{"time": pd.Timestamp.now()}])

        self.df_path: str = os.path.join(self.log_path, df_file_name)
        self.model_path: str = os.path.join(self.log_path, model_file_name)

        self.tensorboard_logdir: Optional[str] = tensorboard_logdir
        self.tb_writer: Optional[SummaryWriter] = None
        if self.tensorboard_logdir is not None:
            self.tb_writer = SummaryWriter(log_dir=tensorboard_logdir)
        
    def update(self, loss: float, silent: Optional[bool] = None) -> None:
        """
        If loss improved, save the model
        """
        if (self.min_loss is None) or (loss < self.min_loss):
            if silent is None:
                silent = self.silent
            torch.save(self.model.state_dict(), self.model_path)
            self.min_loss = loss
            if not silent:
                print(f"Saved better model at {self.log_path} at {pd.Timestamp.now().strftime('%Y-%m-%d --- %H:%M:%S')}")
            
    def log_results(self, data: Dict[str, Any]) -> None:
        """
        Save the provided data to the ModelCheckpoint dataframe 
        """
        # Adds data to the dictionnary
        cols: List[str] = ["time", *data.keys()]
        vals: List[str] = [pd.Timestamp.now(), *data.values()]
        # Index of current step
        idx: int = len(self.df)
        self.df.loc[idx, cols] = vals

        # Saves new dataframe
        self.df.to_csv(self.df_path, index=False, sep=";")

        # Tensorboard
        if self.tensorboard_logdir is not None:
            for col, val in data.items():
                self.tb_writer.add_scalar(f"metrics/{col}", val, idx)
            

    @staticmethod
    def __generate_log_dir(
        log_folder: str = "logs",
        name_template: str = "log_%",
        time_format: str = "%Y_%m_%d__%H_%M_%S__%f",
        ):
        """
        Generate a log directory for storing Pytorch logs
        The logs will be generated from the date of launching the code.
        """
        try:
            date_str: str = pd.Timestamp.now().strftime(time_format)
        except:
            print(f"Wrong time format in log directory creation: {time_format}")
            raise NameError

        directory_name: str = os.path.join(log_folder, name_template.replace("%", date_str))

        # If top log directory doesn't exist: create it
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        # Create log directory
        try:
            os.mkdir(directory_name)
            print(f"Succesfully created log directory: {directory_name}")
        except:
            print(f"Can't create log directory: {directory_name}.\nMaybe it already exists ?")
        return directory_name


def compute_mean_std(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean & std of data in the loader"""
    # Compute the mean over minibatches
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    # Compute the std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img)**2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    # Set the variance of pixels with no variance to 1
    # Because there is no variance
    # these pixels will anyway have no impact on the final decision
    std_img[std_img == 0] = 1

    return mean_img, std_img