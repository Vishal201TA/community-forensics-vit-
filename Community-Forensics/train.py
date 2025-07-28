import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import os, sys
import logging
import random
import utils as ut
import models
import wandb
import gc

from torch.nn.parallel import DistributedDataParallel as DDP

logger: logging.Logger = ut.logger

def keep_only_topn_checkpoints(ckpt_path, keep_top_n):
    ckpt_dir = os.path.dirname(ckpt_path)

    if not ckpt_dir or not os.path.exists(ckpt_dir):
        logger.warning(f"[WARNING] Checkpoint directory does not exist: {ckpt_dir}")
        return  # Exit safely

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    ckpt_files = sorted(ckpt_files, key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))

    while len(ckpt_files) > keep_top_n:
        file_to_delete = ckpt_files.pop(0)
        os.remove(os.path.join(ckpt_dir, file_to_delete))
        logger.info(f"[INFO] Removed old checkpoint: {file_to_delete}")


def train(
        args=None,
        logger=None,
):
    # Setup distributed environment
    rank, local_rank, world_size = ut.dist_setup()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size

    # Initialize W&B if required
    if rank == 0 and args.wandb_token != "":
        ut.init_wandb(args)

    # Load data
    trainLoader, valLoader = ut.get_dataloader(args, mode='train')
    args = ut.get_epochs_for_itrs(args, len(trainLoader))
    trainLoaderLen = len(trainLoader)

    # Load model
    try:
        model = models.ViTClassifier(args=args, device=local_rank, dtype=torch.float32).to(local_rank)
    except Exception as e:
        logger.error(f"[RANK {rank}] Error loading model: {e}")
        sys.exit(1)

    # Set optimizer, scaler, scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp) if args.use_amp else None

    warmup_steps = round(args.warmup_frac * ut.get_total_itrs(args, trainLoaderLen)) if args.warmup_frac > 0 \
        else round(args.warmup_epochs * trainLoaderLen)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * trainLoaderLen - warmup_steps,
        eta_min=ut.get_min_lr(args)
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Load checkpoint or weights
    if args.ckpt_path != '':
        if args.only_load_model_weights:
            model = ut.load_only_weights(model, args.ckpt_path, rank)
            logger.info(f"[RANK {rank}] Fine-tuning from pretrained weights: {args.ckpt_path}")
            epoch_start = 0
            total_itr = 0
        else:
            model, optimizer, scheduler, epoch_start, total_itr = ut.load_checkpoint(
                model, optimizer, scheduler, scaler, args.ckpt_path, rank)
            logger.info(f"[RANK {rank}] Resuming full training from checkpoint: {args.ckpt_path}")
            epoch_start += 1
    else:
        logger.info(f"[RANK {rank}] Training from scratch.")
        epoch_start = 0
        total_itr = 0

    # Optional: Compile model
    try:
        model = torch.compile(model, dynamic=True)
    except Exception as e:
        logger.warning(f"[RANK {rank}] Model compile skipped: {e}")

    logger.info(f"[RANK {rank}] Model ready. Beginning training...")

    # Training loop
    local_window_loss = ut.LocalWindow(100)
    last_epoch = epoch_start - 1
    for epoch in range(epoch_start, args.epochs):
        last_epoch = epoch
        gc.collect()

        avgTrainLoss, total_itr = ut.train_one_epoch(
            args=args,
            epoch=epoch,
            model=model,
            train_loader=trainLoader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            local_window_loss=local_window_loss,
            warmup_steps=warmup_steps,
            rank=rank,
            itr=total_itr,
        )

        if valLoader is not None:
            valLoss, valAcc, valAP = ut.evaluate_one_epoch(
                args=args,
                epoch=epoch,
                model=model,
                dataloader=valLoader,
                criterion=criterion,
                rank=rank,
                evalName="Val",
                separate_eval=False,
                add_sigmoid=(not args.dont_add_sigmoid),
            )
            wandb_log_dict = {
                "epoch": epoch + 1,
                "Loss/Train": avgTrainLoss,
                "Loss/Val": valLoss,
                "Acc/Val": valAcc,
                "AP/Val": valAP
            }
        else:
            valLoss, valAcc, valAP = -1, -1, -1
            wandb_log_dict = {
                "epoch": epoch + 1,
                "Loss/Train": avgTrainLoss
            }

        scheduler.step()

        if rank == 0 and args.wandb_token != "":
            wandb.log(wandb_log_dict, commit=False)

        gc.collect()

    # Save final model and checkpoint
    if rank == 0:
        torch.save(model.state_dict(), args.save_path)
        ut.save_checkpoint(model, optimizer, scheduler, scaler, last_epoch, total_itr,
                           args.save_path.replace('.pt', '_ckpt.pt'))

        if args.ckpt_keep_count > 0 and args.save_path.strip() != "":
            ut.keep_only_topn_checkpoints(args.save_path, args.ckpt_keep_count)


def main():
    args = ut.parse_args()
    args.random_port_offset = np.random.randint(-1000,1000) # randomize to avoid port conflict in same device

    args.resume = "/kaggle/input/deepfake/pytorch/default/1/model_v11_ViT_384_base_ckpt.pt"
    args.save_path = "cifake-finetuned.pt"
    args.epochs = 5
    args.verbose = 2
    args.gpus_list = "0"
    args.cpus_per_gpu = 2

    if args.gpus_list != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_list
        logger.info(f"Setting CUDA_VISIBLE_DEVICES to {args.gpus_list}.")
        args.gpus = len(args.gpus_list.split(','))

    assert args.gpus <= torch.cuda.device_count(), f'Not enough GPUs! {torch.cuda.device_count()} available, {args.gpus} required.'
    assert args.gpus > 0, f'Number of GPUs must be greater than 0!'
    assert args.cpus_per_gpu > 0, f'Number of CPUs per GPU must be greater than 0!'
    
    # Ensure args.save_path is valid
    if not args.save_path or args.save_path.strip() == '':
        args.save_path = './checkpoints/cifake-finetuned.pt'
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        logger.info(f"[INFO] Defaulted save_path to: {args.save_path}")

    # Same for checkpoint save path
    if not args.ckpt_save_path or args.ckpt_save_path.strip() == '':
        args.ckpt_save_path = args.save_path

    

    logger.info(f"Spawning processes on {args.gpus} GPUs. KK")
    logger.info(f"Verbosity: {args.verbose} (0: None, 1: Every epoch, 2: Every iteration)")

    logger.info(f"Model save name: {os.path.basename(args.save_path)}")

    train(
        args=args,
        logger=logger,
    )

if __name__ == "__main__":
    main()