"""
train.py
Style and Structure adapted from https://github.com/meetps/pytorch-semseg

TODO: Move model state loading to a new function
TODO: Save model state on Keyboard Interrupt
TODO: Store colored predictions in tf.summary for vizualisation
TODO: Use progress bars for vizualising training and overall progress
"""

# Standard imports
import os
import time
import shutil
import argparse
from datetime import datetime

# Third party imports
from yaml import load, Loader
import torch
import numpy as np
from PIL import Image
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

    return {"model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "criterion": criterion}


def train(utils, train_loader, val_loader, components_dict):
    "Training"
    cfg = utils["cfg"]
    device = utils.get("device", "cpu")
    logger = utils["logger"]
    writer = utils["writer"]
    # Load components from dict
    model = components_dict["model"]
    optimizer = components_dict["optimizer"]
    scheduler = components_dict["scheduler"]
    criterion = components_dict["criterion"]

    start_epoch = 0
    best_iou = -100.0
    # Load checkpoint if exists
    checkpoint_path = cfg["training"]["checkpoint"]
    if cfg["training"]["resume"] and checkpoint_path is not None:
        if os.path.isfile(checkpoint_path):
            logger.info("Loading model and optimizer from checkpoint: '%s'",
                        checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"]
            best_iou = checkpoint["best_iou"]
        else:
            logger.info("Checkpoint doesn't exist at '%s'", checkpoint_path)

    # Start training
    running_metrics_val = RunningScore(cfg["model"]["n_classes"])
    val_loss_meter = AverageMeter()
    time_meter = AverageMeter()

    epoch = start_epoch
    training_finished = False
    n_batches = len(train_loader)

    while epoch <= cfg["training"]["train_epochs"] and not training_finished:
        epoch += 1
        for batch_num, (images, labels) in enumerate(train_loader, 1):
            current_iter = epoch * n_batches + batch_num
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
            if batch_num % cfg["training"]["print_interval"] == 0:
                # Format a string with stats
                fmt_str = "[Epoch {:03d}][{:03d}/{:03d}] \
Loss: {:0.4f} Time/Image: {:0.4f}"
                print_str = fmt_str.format(
                    epoch,
                    batch_num,
                    n_batches,
                    loss.item(),
                    time_meter.mean/cfg["training"]["batch_size"]
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

            if batch_num % cfg["training"]["val_interval"] == 0 or (
                epoch == cfg["training"]["train_epochs"] and
                    batch_num == n_batches
                    ):
                # Set the model to evaluation mode
                model.eval()
                tqdm_val_loader = tqdm(enumerate(val_loader))
                with torch.no_grad():
                    for b_num_val, (images_val, labels_val) in tqdm_val_loader:
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs_val = model(images_val)
                        val_loss = criterion(outputs_val, labels_val)

                        predictions = outputs_val.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, predictions)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss",
                                  val_loss_meter.mean, current_iter)
                logger.info("[VALID][Epoch %03d][%03d/%03d] Loss: %.04f",
                            epoch, batch_num, n_batches,
                            val_loss_meter.mean)

                score, class_iu = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(f"{k.ljust(15)}: {v}")
                    logger.info("{%s}: {%f}", k.ljust(15), v)
                    writer.add_scalar(f"val_metrics/{k.strip()}",
                                      v, current_iter)

                for k, v in class_iu.items():
                    logger.info("[Class IoU] {%d}: {%f}", k, v)
                    writer.add_scalar(f"val_metrics/class_iou/class_{k}",
                                      v, current_iter)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU"] >= best_iou:
                    best_iou = score["Mean IoU"]
                    state = {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        f"{cfg['model']['architecture']}_best_model.pkl"
                    )
                    torch.save(state, save_path)

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

    # Get color mapper
    colored = get_color_mapper(cfg["data"])
    utils = {"cfg": cfg,
             "device": device,
             "logger": logger,
             "writer": writer,
             "colored": colored}
    train_loader, val_loader = prepare_data(utils)
    components = prepare_components(utils)
    train(utils, train_loader, val_loader, components)
