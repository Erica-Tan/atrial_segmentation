import os
import argparse
import wandb
from functools import partial
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monai import losses
from monai.inferers import SliceInferer, SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from ptflops import get_model_complexity_info

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader
from utils.network_utils import get_model
from utils.utils import (
    AsDiscrete,
    print_trainable_parameters,
    freeze_model_weights
)


parser = argparse.ArgumentParser(description="Segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--data_dir", default="./data/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")

parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--model_name", default="resunet", type=str, help="model name")
parser.add_argument("--lrschedule", default="cosine_anneal", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--loss", default="diceceloss", type=str, help="loss function")
parser.add_argument("--save_weight_update_plot", action="store_true", help="save weight update magnitude plot")
parser.add_argument('--dataset', type=str, default='utah', help='dataset name')
parser.add_argument("--not_include_background", action="store_true", help="include background in loss and evaluation matrix")
parser.add_argument(
    '--semantic_classes', nargs='+', required=False, default=['Dice_Val_RA_LA_wall', 'Dice_Val_RA_endo', 'Dice_Val_LA_endo'],
                        help="List of classes. Default: ['Dice_Val_RA_LA_wall', 'Dice_Val_RA_endo', 'Dice_Val_LA_endo']"
)

parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=4, type=int, help="number of output channels")
parser.add_argument("--roi_x", default=272, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=272, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=272, type=int, help="roi size in z direction")
parser.add_argument("--spatial_dims", default=2, type=int, help="spatial dimension of input data")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--deep_supervision", action="store_true", help="use deep_supervision")
parser.add_argument("--transform", default='randomcrop', type=str)
parser.add_argument("--act", default="prelu", type=str, help="activation layer type")

parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist_url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")

parser.add_argument("--wandb", action="store_true", help="use wandb")
parser.add_argument("--experiment_name", default='atrial_segmentation', type=str, help="experiment name")
parser.add_argument("--run_name", default='test', type=str, help="run name")
parser.add_argument("--profiler", action="store_true", help="use profiler")
parser.add_argument("--profiler_dir", default="./wandb/latest-run/tbprofile", type=str, help="directory to save profiler logs")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.include_background = not args.not_include_background
    args.logdir = os.path.normpath("./runs/" + args.logdir)
    os.makedirs(args.logdir, exist_ok=True)

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)

    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    if args.rank == 0:
        if args.wandb:
            wandb.init(project=args.experiment_name, name=args.run_name, config=vars(args), sync_tensorboard=True)

    args.test_mode = False
    train_loader, val_loader = get_loader(args)

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir

    # Load model
    model = get_model(args)

    # Monai unetplusplus will return list regardless if deep supervision is true or not
    # need to set it to true after loading the model
    if args.model_name == 'unetplusplus':
        args.deep_supervision = True

    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name), map_location="cpu")["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    if args.use_ssl_pretrained:
        try:
            if args.model_name == 'swinunetr':
                model_dict = torch.load("./pretrained_models/model_swinvit.pt")
                state_dict = model_dict["state_dict"]
                # fix potential differences in state dict keys from pre-training to
                # fine-tuning
                if "module." in list(state_dict.keys())[0]:
                    print("Tag 'module.' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("module.", "swinViT.")] = state_dict.pop(key)
            else:
                state_dict = torch.load("./pretrained_models/UNETR_model_best_acc.pth")
                for key in list(state_dict.keys()):
                    if 'mlp.linear' in key:
                        state_dict.pop(key)
                    if 'decoder' in key:
                        state_dict.pop(key)
                    if 'out.conv.conv' in key:
                        state_dict.pop(key)
                state_dict.pop('vit.patch_embedding.position_embeddings')
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            print("Using pretrained self-supervised backbone weights !")

            # freeze parameters from the pretrain weight
            model = freeze_model_weights(model, state_dict.keys())

        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    # Print model parameters
    print_trainable_parameters(model)

    # Calculate FLOP
    input_shape = tuple(next(iter(train_loader))['image'].shape)[1:]
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    if args.loss == "diceceloss":
        dice_loss = losses.DiceCELoss(
            include_background=args.include_background, to_onehot_y=False, softmax=True, squared_pred=True,
            smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    elif args.loss == "dicefocalloss":
        dice_loss = losses.DiceFocalLoss(
            include_background=args.include_background, to_onehot_y=False, softmax=True, squared_pred=True,
            smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    elif args.loss == "generalizeddiceceloss":
        dice_loss = losses.GeneralizedDiceLoss(
            include_background=args.include_background, to_onehot_y=False, softmax=True, squared_pred=True,
            smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    elif args.loss == "dice":
        dice_loss = losses.DiceLoss(
            include_background=args.include_background, to_onehot_y=False, softmax=True, squared_pred=True,
            smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        raise ValueError("Unsupported Optimization loss: " + str(args.loss))

    dice_acc = DiceMetric(include_background=args.include_background, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    # post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)

    if args.spatial_dims == 3:
        slice_infer = SlidingWindowInferer(
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size,
            overlap=args.infer_overlap,
        )
    else:
        slice_infer = SliceInferer(
            roi_size=(args.roi_x, args.roi_y),
            sw_batch_size=args.sw_batch_size,
            cval=-1,
            spatial_dim=2,
            progress=False,
            overlap=args.infer_overlap
        )
    model_inferer = partial(slice_infer, network=model)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    # non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.rank == 0:
        Optimizer_metadata = {}
        for ind, param_group in enumerate(optimizer.param_groups):
            Optimizer_metadata[f"optimizer_param_group_{ind}"] = {
                key: value for (key, value) in param_group.items() if "params" not in key
            }

        if args.wandb:
            for key, value in Optimizer_metadata.items():
                wandb.config.key = value

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    if args.profiler:
        profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profiler_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        )
    else:
        profiler = None

    accuracy = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_acc=best_acc,
        post_label=None,
        post_pred=post_pred,
        semantic_classes=args.semantic_classes,
        wandb=wandb if args.wandb else None,
        profiler=profiler if args.profiler else None
    )

    if args.rank == 0:
        if args.wandb:
            wandb.finish()

    return accuracy


if __name__ == "__main__":
    main()
