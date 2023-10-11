"""
utils.py
Implements needed utilities
"""

import os
import copy
import logging
from datetime import datetime

import torch
import numpy as np
from torch import optim

from models import Unet
from loss import cross_entropy_2d
from schedulers import ConstantLR
from labels import id2trainId, trainId2color


# Cityscapes metrics
MEAN = np.array([72.55410438, 81.93415236, 71.4297832]) / 255
STD = np.array([51.04788791, 51.76003371, 50.94766331]) / 255

# Imagenet metrics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# No normalisation metrics
ZERO_MEAN = np.array([0.0, 0.0, 0.0])
UNIT_STD = np.array([1.0, 1.0, 1.0])

LOG_NAME = "cityscapes_logs"
logger = logging.getLogger(LOG_NAME)


def get_logger(logdir):
    "creates a logger, exports logs to <logdir>/<timestamp>"
    logger_ = logging.getLogger(LOG_NAME)
    file_path = os.path.join(
        logdir,
        "logs_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.FileHandler(file_path)
    handler.setFormatter(formatter)
    logger_.addHandler(handler)
    logger_.setLevel(logging.INFO)
    return logger_


def get_model(model_data):
    "Returns a model instance"
    # An inner function to retrieve model class
    def _get_model_class(name):
        model_map = {
            "unet": Unet,
        }
        return model_map.get(name.lower())

    # Read the model name from config.model.architecture
    name = model_data["architecture"]
    # Read the number of classes from config.model.n_classes
    n_classes = model_data["n_classes"]
    # Make a copy of parameter dictionary
    param_dict = copy.deepcopy(model_data)
    # Delete the used items
    del param_dict["architecture"]
    del param_dict["n_classes"]
    # Obtain the model class using name
    model_class = _get_model_class(name)
    # Raise an error if model is not available
    if model_class is None:
        raise f"Model '{name}' is not available!"
    # Create an instance of model with given params (if any)
    model_instance = model_class(n_classes=n_classes, **param_dict)
    params_str = ", ".join([f"{k}: {v}" for k, v in param_dict.items()])
    logger.info("Using model: %s (%s)", model_class.__name__, params_str)
    return model_instance


def get_optimizer(optimizer_data):
    "Returns an optimizer class"
    def _get_optim_class(name):
        optim_map = {
            "sgd": optim.SGD,
            "adam": optim.Adam,
            "nadam": optim.NAdam,
            "adamax": optim.Adamax,
            "adagrad": optim.Adagrad,
            "rmsprop": optim.RMSprop
        }
        return optim_map.get(name.lower())

    name = optimizer_data.get("name", "adam")
    param_dict = copy.deepcopy(optimizer_data)
    del param_dict["name"]
    optimizer_class = _get_optim_class(name)
    if optimizer_class is None:
        raise f"Optimizer '{name}' is not available!"

    return optimizer_class


def get_scheduler(optimizer, scheduler_data):
    "Returns scheduler instance"
    def _get_scheduler_class(name):
        scheduler_map = {
            "constant": ConstantLR
        }
        return scheduler_map.get(name.lower())

    name = scheduler_data.get("name", "constant")
    param_dict = copy.deepcopy(scheduler_data)
    del param_dict["name"]
    scheduler_class = _get_scheduler_class(name)
    if scheduler_class is None:
        raise f"Scheduler '{name}' is not available!"
    # Create an instance of scheduler with given params (if any)
    scheduler_instance = scheduler_class(optimizer=optimizer, **param_dict)
    params_str = ", ".join([f"{k}: {v}" for k, v in param_dict.items()])
    logger.info("Using Scheduler: %s (%s)",
                scheduler_class.__name__, params_str)
    return scheduler_instance


def get_loss(loss_data):
    "Returns loss function"
    def _get_loss_function(name):
        loss_map = {
            "ce2d": cross_entropy_2d
        }
        return loss_map.get(name.lower())

    name = loss_data.get("name", "ce2d")
    logger.info("Using Loss: %s", name)
    loss = _get_loss_function(name)
    return loss


def preprocess_labels(mask):
    """ Map IDs to train ids for semantic masks
        Input: PIL image
        Output: tensor
    """
    mask_np = np.array(mask)
    mask_np_clone = mask_np.copy()
    for id_, train_id_ in id2trainId.items():
        mask_np_clone[mask_np == id_] = train_id_

    mapped_tensor = torch.tensor(mask_np_clone, dtype=torch.int64)
    return mapped_tensor


def get_color_mapper(dim_data):
    "returns closure that maps label to RGB colors"
    height = dim_data["img_height"]
    width = dim_data["img_width"]
    n_channels = 3

    def colored(pred):
        rgb_pred = torch.zeros((height, width, n_channels)).long()
        for (train_id, color) in trainId2color.items():
            rgb_pred[pred == train_id] = torch.tensor(color)

        return rgb_pred
    return colored
