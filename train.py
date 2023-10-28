"""
train.py
Style adapted from https://github.com/meetps/pytorch-semseg

TODO: Save model state on Keyboard Interrupt
TODO: Implement early stopping
TODO: Use progress bars for vizualising training and overall progress
"""

# Standard imports
import os
import sys
import time
import shutil
import argparse
from datetime import datetime

# Third party imports
from yaml import load, Loader
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local imports
from dataset import Cityscapes
from metrics import AverageMeter, RunningScore
from utils import (ZERO_MEAN, UNIT_STD,
                   preprocess_labels,
                   get_logger,
                   get_model,
                   get_optimizer,
                   get_scheduler,
                   get_loss,
                   get_color_mapper)


def prepare_data(utils):
    "Setup datasets and dataloaders"
    cfg = utils["cfg"]
    logger = utils["logger"]
    image_size = (cfg["data"]["img_height"], cfg["data"]["img_width"])
    # Transformations for training data
    train_transforms = transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=ZERO_MEAN, std=UNIT_STD),
    ])

    # Transformations for evaluation (val/test) data
    eval_transforms = transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=ZERO_MEAN, std=UNIT_STD),
    ])

    # Transformations for all annotation
    annot_transforms = transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(preprocess_labels),
    ])

    # Creating instances of datasets
    data_path = cfg["data"]["root_path"]
    train_dataset = Cityscapes(root_dir=data_path,
                               split=cfg["data"]["train_split"],
                               target_type="semantic",
                               image_transforms=train_transforms,
                               mask_transforms=annot_transforms)

    val_dataset = Cityscapes(root_dir=data_path,
                             split=cfg["data"]["val_split"],
                             target_type="semantic",
                             image_transforms=eval_transforms,
                             mask_transforms=annot_transforms)

    # Setup dataloaders
    t_loader = DataLoader(train_dataset,
                          batch_size=cfg["training"]["batch_size"],
                          num_workers=cfg["training"]["n_workers"],
                          shuffle=True)

    v_loader = DataLoader(val_dataset,
                          batch_size=cfg["training"]["batch_size"],
                          num_workers=cfg["training"]["n_workers"])

    logger.info("Dataloaders created!")
    return t_loader, v_loader


def prepare_components(utils):
    "A function to setup model, optimizer, scheduler and loss function"
    cfg = utils["cfg"]
    device = utils.get("device", "cpu")
    logger = utils["logger"]
    # Model
    model = get_model(cfg["model"]).to(device)
    model = torch.nn.DataParallel(model,
                                  device_ids=range(torch.cuda.device_count()))

    # Optimizer
    optimizer_class = get_optimizer(cfg["training"]["optimizer"])
    param_dict = {k: v for k, v in cfg["training"]["optimizer"].items()}
    del param_dict["name"]

    optimizer = optimizer_class(model.parameters(), **param_dict)
    params_str = ", ".join([f"{k}: {v}" for k, v in param_dict.items()])
    logger.info("Using optimizer: %s (%s)",
                optimizer_class.__name__, params_str)

    # LR Scheduler
    scheduler = get_scheduler(optimizer, cfg["training"]["scheduler"])

    # Loss function
    criterion = get_loss(cfg["training"]["loss"])

    # Start Epoch
    start_epoch = 0
    best_iou = -100.0

    return {"model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "criterion": criterion,
            "start_epoch": start_epoch,
            "best_iou": best_iou}


def load_state(utils, components):
    "Load model, optimizer, scheduler state from previous checkpoint"
    cfg = utils["cfg"]
    logger = utils["logger"]

    checkpoint_path = cfg["training"]["checkpoint"]
    if os.path.isfile(checkpoint_path):
        logger.info("Loading model and optimizer from checkpoint: '%s'",
                    checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        components["model"].load_state_dict(checkpoint["model_state"])
        components["optimizer"].load_state_dict(checkpoint["optimizer_state"])
        components["scheduler"].load_state_dict(checkpoint["scheduler_state"])
        components["start_epoch"] = checkpoint["epoch"]
        components["best_iou"] = checkpoint["best_iou"]
    else:
        logger.info("Checkpoint doesn't exist at '%s'", checkpoint_path)

    return components


def save_state(utils, components, iou, name=None):
    "Save model, and other components"
    cfg = utils["cfg"]
    writer = utils["writer"]
    if name is None:
        name = cfg['model']['architecture'] + "_best_model"
    state = {
        "epoch": components["epoch"],
        "model_state": components["model"].state_dict(),
        "optimizer_state": components["optimizer"].state_dict(),
        "scheduler_state": components["scheduler"].state_dict(),
        "best_iou": iou,
    }
    save_path = os.path.join(
        writer.file_writer.get_logdir(),
        f"{name}.pkl"
    )
    torch.save(state, save_path)


def train(utils, train_loader, val_loader, components):
    """ Training
        utils is expected to be a dictionary with
        cfg: config from yaml
        device: device (default CPU)
        logger: logger used
        writer: summary writer used
    """
    cfg = utils["cfg"]
    device = utils.get("device", "cpu")
    logger = utils["logger"]
    writer = utils["writer"]
    colored = utils["colored"]

    checkpoint_path = cfg["training"]["checkpoint"]
    if cfg["training"]["resume"] and checkpoint_path is not None:
        components = load_state(utils, components)

    # Load components from dict
    model = components["model"]
    optimizer = components["optimizer"]
    scheduler = components["scheduler"]
    criterion = components["criterion"]
    start_epoch = components["start_epoch"]
    best_iou = components["best_iou"]

    # Start training
    running_metrics_val = RunningScore(cfg["model"]["n_classes"])
    val_loss_meter = AverageMeter()
    time_meter = AverageMeter()

    epoch = start_epoch
    training_finished = False
    n_batches = len(train_loader)
    n_epochs = cfg["training"]["train_epochs"]
    batch_size = cfg["training"]["batch_size"]

    while epoch <= n_epochs and not training_finished:
        epoch += 1
        for batch_num, (images, labels) in enumerate(train_loader, 1):
            current_iter = (epoch - 1) * n_batches + batch_num
            # Mark current time
            tic = time.time()
            # Set the model to train mode
            model.train()

            # Copy images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Set the gradients to zero
            optimizer.zero_grad()
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            # Compute the gradients
            loss.backward()
            # Step the optimizer
            optimizer.step()
            # Step the scheduler
            scheduler.step()
            # Mark current time
            toc = time.time()
            # Store the time taken
            time_meter.update(toc - tic)

            # If batch reaches print interval
            print_interval = cfg["training"]["print_interval"]
            val_interval = cfg["training"]["val_interval"]
            if batch_num % print_interval == 0:
                time_per_batch = time_meter.mean
                n_batches_left = n_batches - batch_num
                n_vals_left = ((n_batches - batch_num) // val_interval) + 1
                eta_epoch = time_per_batch * (n_batches_left + n_vals_left)

                # Format a string with stats
                fmt_str = "[{:03d}/{:03d}]\x1B[36m[{:03d}/{:03d}] \
\x1B[31mLoss: {:0.3f}\x1B[0m Time/Batch({}): {:0.2f}s \
\x1B[33mETA: {:0.2f}m\x1B[0m"
                print_str = fmt_str.format(
                    epoch, n_epochs,
                    batch_num, n_batches,
                    loss.item(), batch_size,
                    time_per_batch, eta_epoch/60
                )
                # Print it
                print(print_str)
                # Log it
                logger.info(print_str)
                # Record it
                writer.add_scalar(f"loss/epochs/epoch {epoch}",
                                  loss.item(), batch_num)
                writer.add_scalar("loss/train_loss",
                                  loss.item(), current_iter)
                # Reset time meter
                time_meter.reset()

            if batch_num % val_interval == 0 or (
                batch_num == n_batches) or (
                epoch == cfg["training"]["train_epochs"] and
                    batch_num == n_batches
                    ):
                # Set the model to evaluation mode
                model.eval()
                tqdm_val_loader = tqdm(enumerate(val_loader))
                # Don't compute the gradients for validation
                with torch.no_grad():
                    for b_num_val, (images_val, labels_val) in tqdm_val_loader:
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs_val = model(images_val)
                        val_loss = criterion(outputs_val, labels_val)

                        # Convert the onehot encoded feature maps to prediction
                        predictions = outputs_val.data.max(1)[1]
                        gt = labels_val.data

                        running_metrics_val.update(gt, predictions)
                        val_loss_meter.update(val_loss.item())

                # Register and log val loss
                writer.add_scalar("loss/val_loss",
                                  val_loss_meter.mean, current_iter)
                logger.info("[VALID][Epoch %03d][%03d/%03d] Loss: %.04f",
                            epoch, batch_num, n_batches,
                            val_loss_meter.mean)

                # Register and log metrics
                score, class_iu = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(f"\x1B[32m{k.ljust(15)}: {v:0.4f}\x1B[0m")
                    logger.info("%s: %f", k.ljust(15), v)
                    writer.add_scalar(f"val_metrics/{k.strip()}",
                                      v, current_iter)

                # Register and log class wise IoU scores
                for k, v in class_iu.items():
                    logger.info("[Class IoU] %s: %f", k, v)
                    writer.add_scalar(f"val_metrics/class_iou/{k}",
                                      v, current_iter)

                val_loss_meter.reset()
                running_metrics_val.reset()

                # map predictions to colors
                rgb_preds = torch.stack(list(map(colored, predictions)))
                rgb_labels = torch.stack(list(map(colored, labels_val)))
                # Write images and label masks only once
                if current_iter == val_interval:
                    writer.add_images("Images", images_val, current_iter)
                    writer.add_images("Labels", rgb_labels, current_iter)

                writer.add_images("Predictions", rgb_preds, current_iter)
                # Save the model and components for best IoU score
                if score["Mean IoU"] >= best_iou:
                    components_ = {
                        "epoch": epoch,
                        "model": model,
                        "optimizer": optimizer,
                        "scheduler": scheduler
                    }
                    save_state(utils, components_, score["Mean IoU"])

        print()
        if epoch == cfg["training"]["train_epochs"]:
            training_finished = True
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="config.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = load(f, Loader=Loader)

    run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logdir = os.path.join("runs", run_id)
    writer = SummaryWriter(logdir)

    print(f"RUNDIR: {logdir}")
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Initiated...")

    # Setup seeds for reproductibility
    torch.manual_seed(cfg.get("seed", 1024))
    torch.cuda.manual_seed(cfg.get("seed", 1024))
    np.random.seed(cfg.get("seed", 1024))

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)
    print(f"Using device: {device}")

    # Get color mapper
    colored = get_color_mapper(cfg["data"])
    utils = {"cfg": cfg,
             "device": device,
             "logger": logger,
             "writer": writer,
             "colored": colored}
    train_loader, val_loader = prepare_data(utils)
    components = prepare_components(utils)
    try:
        train(utils, train_loader, val_loader, components)
    except KeyboardInterrupt:
        writer.close()
        print("*Keyboard Interrupt*")
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
