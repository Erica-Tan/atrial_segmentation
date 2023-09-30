import os
import shutil
import time
from tqdm import tqdm
import numpy as np
import glob
from monai.data import decollate_batch
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast

from utils.utils import (
    distributed_all_gather,
    AverageMeter,
    save_weight_update_plot
)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    run_loss = AverageMeter()

    pbar = tqdm(loader, desc="Training (Epoch X / X) (loss=X.X)", dynamic_ncols=True)
    for idx, batch_data in enumerate(pbar):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)

        optimizer.zero_grad()

        with autocast(enabled=args.amp):
            if args.model_name == 'diffunet':
                x_start = target
                x_start = (x_start) * 2 - 1
                x_t, t, noise = model(x=x_start, pred_type="q_sample")
                logits = model(x=x_t, step=t, image=data, pred_type="denoise")
            else:
                logits = model(data)

            if args.deep_supervision:
                loss = 0
                for logit in logits:
                    loss += loss_func(logit, target)
                loss /= len(logits)
            else:
                loss = loss_func(logits, target)

            if args.model_name == 'diffunet':
                mse = nn.MSELoss()
                loss_mse = mse(torch.softmax(logits, 1), target)
                loss += loss_mse

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        pbar.set_description("Training %d (Epoch %d / %d) (loss=%3f)" % (args.rank, epoch, args.max_epochs-1, run_loss.avg))

    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validate (Epoch X) (dice=X.X)", dynamic_ncols=True)
        for idx, batch_data in enumerate(pbar):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    kwargs = {"pred_type": "ddim_sample"} if args.model_name == 'diffunet' else {}
                    logits = model_inferer(data, **kwargs)
                else:
                    logits = model(data)

            val_labels_list = decollate_batch(target)
            # val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            if args.deep_supervision:
                val_outputs_list = decollate_batch(logits[-1])
            else:
                val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            pbar.set_description("Validate %d (Epoch %d) (dice=%3f)" % (args.rank, epoch, np.mean(run_acc.avg)))

    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    best_acc=0.0,
    post_label=None,
    post_pred=None,
    semantic_classes=None,
    wandb=None,
    profiler=None
):
    scaler = None
    if args.amp:
        scaler = GradScaler()

    ud = []
    val_acc_max = best_acc
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )

        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "lr: {:.6f}".format(optimizer.param_groups[0]["lr"]),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and wandb is not None:
            wandb.log({"train/train_loss": train_loss, "train/lr": optimizer.param_groups[0]["lr"]})

        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
                model_inferer=model_inferer,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_acc)

            if args.rank == 0:
                val_results = "Final validation  {}/{}, mean dice {}".format(epoch, args.max_epochs - 1, val_avg_acc)
                val_metrics = {"val/mean_dice": np.mean(val_acc),}
                if semantic_classes is not None:
                    start_idx = 1 if len(semantic_classes) < len(val_acc) else 0
                    for val_channel_ind in range(len(semantic_classes)):
                        val_results += f", {semantic_classes[val_channel_ind]} {val_acc[val_channel_ind + start_idx]}"
                        val_metrics[f"val/{semantic_classes[val_channel_ind]}"] = val_acc[val_channel_ind + start_idx]
                print(val_results + "time {:.2f}s".format(time.time() - epoch_time))

                if wandb is not None:
                    wandb.log(val_metrics)

                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True

            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

                    if wandb is not None:
                        artifact_name = f"{wandb.run.id}_context_model"
                        at = wandb.Artifact(artifact_name, type="model")
                        at.add_file(os.path.join(args.logdir, "model.pt"))
                        wandb.log_artifact(at, aliases=[f"epoch_{epoch}"])

        if args.rank == 0 and args.save_weight_update_plot:
            lr = optimizer.param_groups[0]["lr"]
            with torch.no_grad():
                ud.append([((lr * p.grad).std() / p.data.std()).log10().item() for p in model.parameters()])

        if scheduler is not None:
            scheduler.step()

        if args.rank == 0 and profiler is not None:
            profiler.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    if args.rank == 0 and args.wandb:
        if args.save_weight_update_plot:
            # Weight update / weight magnitude plot
            plt = save_weight_update_plot(model.parameters(), ud, os.path.join(args.logdir, "weight_update.jpg"))
            wandb.log({"Weight update / weight magnitude plot": plt})

        if profiler is not None:
            # profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
            # profile_art.add_file(glob.glob(f"{args.profiler_dir}/*.pt.trace.json")[0], "trace.pt.trace.json")
            # wandb.log_artifact(profile_art)

            # manually upload the PyTorch Profiler JSON file for tensorboard
            wandb.save(glob.glob(f"{args.profiler_dir}/*.pt.trace.json")[0], base_path=f"{'/'.join(list(args.profiler_dir.split('/')[0:-1]))}")

    return val_acc_max
