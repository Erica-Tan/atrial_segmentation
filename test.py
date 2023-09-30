import os
import argparse
import tempfile
from functools import partial
import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import SliceInferer, sliding_window_inference
import wandb
from monai.visualize import blend_images
from matplotlib import pyplot as plt
import time

from utils.data_utils import get_loader
from utils.network_utils import get_model
from utils.utils import (
    ConvertToMultiChannel,
    load_nii
)
from utils.postprocessing_utils import (
    apply_postprocessing,
    get_center_crop_coords
)
from metrics.metrics import (
    dice,
    hausdorff_distance,
    average_surface_distance,
)

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")

parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument('--dataset', type=str, default='utah', help='dataset name')
parser.add_argument("--exp_name", default="test", type=str, help="experiment name")
parser.add_argument("--model_name", default="resunet", type=str, help="model name")

parser.add_argument("--spatial_dims", default=2, type=int, help="spatial dimension of input data")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=4, type=int, help="number of output channels")
parser.add_argument("--roi_x", default=272, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=272, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=272, type=int, help="roi size in z direction")
parser.add_argument("--act", default="prelu", type=str, help="activation layer type")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--transform", default='randomcrop', type=str)
parser.add_argument('--spacing', nargs='+', type=float, default=(0.625, 0.625, 2.5))

parser.add_argument("--wandb", action="store_true", help="use mlflow")
parser.add_argument("--experiment_name", default='evaluation', type=str, help="experiment name")
parser.add_argument("--run_name", default='test', type=str, help="run name")
parser.add_argument('--semantic_classes', nargs='+', required=False, default=['RA_LA_wall', 'RA_endo', 'LA_endo'],
                        help="List of classes. Default: ['RA_LA_wall', 'RA_endo', 'LA_endo']")

parser.add_argument("--postprocessing", action="store_true", help="use largest component postprocessing")


def render(image, label, prediction, show=False, out_file=None, colormap="hsv"):
    """
    Render a two-column overlay, where the first column is the target (correct) label atop the original image,
    and the second column is the predicted label atop the original image.

    Args:
        image: the input image to blend with label and prediction data.
        label: the input label to blend with image data.
        prediction: the predicted label to blend with image data.
        show: whether the figure will be printed out. default to False.
        out_file: directory to save the output figure. if none, no save happens. default to None.
        colormap: desired colormap for the plot. default to `spring`. for more details, please refer to:
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    correct_blend = blend_images(image=image, label=label, alpha=0.5, cmap=colormap, rescale_arrays=True)
    predict_blend = blend_images(image=image, label=prediction, alpha=0.5, cmap=colormap, rescale_arrays=True)

    print(correct_blend.shape, predict_blend.shape)

    lower, rnge = 5, 5
    count = 1
    fig = plt.figure("blend image and label", (8, 4 * rnge))
    for i in range(lower, lower + rnge):
        # plot the slice 50 - 100 of image, label and blend result
        slice_index = 3 * i
        plt.subplot(rnge, 2, count)
        count += 1
        plt.title(f"correct label slice {slice_index}")
        plt.imshow(torch.moveaxis(correct_blend[:, :, :, slice_index], 0, -1))
        plt.subplot(rnge, 2, count)
        count += 1
        plt.title(f"predicted label slice {slice_index}")
        plt.imshow(torch.moveaxis(predict_blend[:, :, :, slice_index], 0, -1))
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()
    return fig


def main():
    args = parser.parse_args()
    args.spacing = tuple(args.spacing)
    args.test_mode = True
    args.distributed = False
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    # logged artifacts will be stored in the local mlruns/ folder, we are using this as a tmp workspace
    overlay_tmp_dir = tempfile.mkdtemp()

    if args.wandb:
        wandb.init(project=args.experiment_name, name=args.run_name, config=vars(args), sync_tensorboard=True)

        for val_channel_ind in range(len(args.semantic_classes)):
            wandb.define_metric(f"{args.semantic_classes[val_channel_ind]}_dice", summary="mean")
            wandb.define_metric(f"{args.semantic_classes[val_channel_ind]}_hd", summary="mean")
            wandb.define_metric(f"{args.semantic_classes[val_channel_ind]}_sd", summary="mean")

        wandb.define_metric("Mean Dice", summary="mean")
        wandb.define_metric("Mean Hausdorff distance", summary="mean")
        wandb.define_metric("Mean Average Surface Distance", summary="mean")

    args.deep_supervision = False

    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        model = get_model(args)
        model_dict = torch.load(pretrained_pth, map_location="cpu")["state_dict"]
        model.load_state_dict(model_dict)
        # logged_model = 'runs:/7bdab1c3e5114d1b94a351526464723f/model_best'
        # model = mlflow.pytorch.load_model(logged_model)
    model.eval()
    model.to(device)

    if args.spatial_dims == 3:
        inf_size = [args.roi_x, args.roi_y, args.roi_z]

        model_inferer_test = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
        )
    else:
        slice_infer = SliceInferer(
            roi_size=(args.roi_x, args.roi_y),
            sw_batch_size=args.sw_batch_size,
            cval=-1,
            spatial_dim=2,
            progress=True,
            overlap=args.infer_overlap
        )
        model_inferer_test = partial(slice_infer, network=model)

    save_visual_every = 2

    infer_time = []

    with torch.no_grad():
        dice_list_case = []
        hd_list_case = []
        sd_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))

            # Get meta data
            if 'image_meta_dict' in batch:
                session = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
                patient = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-3]
            else:
                session = batch['meta_data']["session"][0] if "session" in batch['meta_data'] else None
                patient = batch['meta_data']["patient"][0]

            print(f"Inference on case {patient} {session}, {val_inputs.shape} {val_labels.shape}")

            # start inference
            start_time = time.time()
            val_outputs = model_inferer_test(val_inputs)
            infer_time.append(time.time() - start_time)

            # convert label and predction from [#label, width, height. #slice] -> [width, height. #slice]
            if isinstance(val_outputs, tuple):
                val_outputs = torch.argmax(val_outputs[-1], dim=1).cpu().numpy().astype(np.uint8)
            else:
                val_outputs = torch.argmax(val_outputs, dim=1).cpu().numpy().astype(np.uint8)

            val_labels = torch.argmax(val_labels, dim=1).cpu().numpy().astype(np.uint8)

            # Create overlay image
            if args.wandb and i % save_visual_every == 0:
                print("tracking overlay...", end="")
                render(
                    image=val_inputs[0].cpu(),
                    label=torch.tensor(val_labels),
                    prediction=torch.tensor(val_outputs),
                    out_file=os.path.join(overlay_tmp_dir, f"{patient}_{session}.png"),
                )
                print(os.path.join(overlay_tmp_dir, f"{patient}_{session}.png"))

                profile_art = wandb.Artifact(f"render-{patient}-{session}", type="render")
                profile_art.add_file(os.path.join(overlay_tmp_dir, f"{patient}_{session}.png"), f"{patient}_{session}.png")
                wandb.log_artifact(profile_art)

                print("done.")

            # convert the centercrop results back to the original size
            if args.spatial_dims == 2 and args.transform == "centercrop":
                val_labels = load_nii(batch['meta_data']['label_path'][0])
                val_labels = ConvertToMultiChannel()(image=val_labels, mask=val_labels)['mask']
                val_labels = np.argmax(val_labels, axis=0)

                # get ranges for the images subcrop based on midpoint
                height, width, slices = val_labels.shape
                x1, y1, x2, y2 = get_center_crop_coords(height, width, args.roi_x, args.roi_y)
                full_val_outputs = np.zeros([height, width, slices])
                for j in range(val_outputs[0].shape[2]):
                    full_val_outputs[x1:x2, y1:y2, j] = val_outputs[0][:, :, j]

                val_labels = val_labels[None, :]
                val_outputs = full_val_outputs[None, :]

            val_outputs = val_outputs[0]
            val_labels = val_labels[0]

            if args.postprocessing:
                # remove_largest_component_segmentation
                pp_fn_kwargs = [{'labels_or_regions': [1, 2, 3]}, {'labels_or_regions': 1}, {'labels_or_regions': 2}, {'labels_or_regions': 3}]
                val_outputs = apply_postprocessing(val_outputs, pp_fn_kwargs)

            print(val_outputs.shape, val_labels.shape, args.spacing)

            # compute dice score
            dice_list_sub = []
            for i in range(1, args.out_channels):
                organ_dice = dice(val_outputs == i, val_labels == i)
                print("{}: {:.3f}".format(args.semantic_classes[i - 1], organ_dice))
                if args.wandb:
                    wandb.log({f"{args.semantic_classes[i-1]}_dice": organ_dice})
                dice_list_sub.append(organ_dice)
            if args.wandb:
                wandb.log({f"Mean Dice": np.mean(dice_list_sub)})
            print("Mean Dice: {:.3f}".format(np.mean(dice_list_sub)))
            dice_list_case.append(dice_list_sub)

            # compute Hausdorff distance
            hd_list_sub = []
            for i in range(1, args.out_channels):
                organ_hd = hausdorff_distance(val_outputs == i, val_labels == i, spacing=args.spacing)
                print("{}: {:.3f}".format(args.semantic_classes[i - 1], organ_hd))
                if args.wandb:
                    wandb.log({f"{args.semantic_classes[i-1]}_hd": organ_hd})
                hd_list_sub.append(organ_hd)
            if args.wandb:
                wandb.log({f"Mean Hausdorff distance": np.mean(hd_list_sub)})
            print("Mean Hausdorff distance: {:.3f}".format(np.mean(hd_list_sub)))
            hd_list_case.append(hd_list_sub)

            # compute Average Surface Distance
            sd_list_sub = []
            for i in range(1, args.out_channels):
                organ_sd = average_surface_distance(val_outputs == i, val_labels == i, spacing=args.spacing)
                print("{}: {:.3f}".format(args.semantic_classes[i-1], organ_sd))
                if args.wandb:
                    wandb.log({f"{args.semantic_classes[i-1]}_sd": organ_sd})
                sd_list_sub.append(organ_sd)
            if args.wandb:
                wandb.log({f"Mean Average Surface Distance": np.mean(sd_list_sub)})
            print("Mean Average Surface Distance: {:.3f}".format(np.mean(sd_list_sub)))
            sd_list_case.append(sd_list_sub)

            # save image
            ni_img = sitk.GetImageFromArray(val_outputs)
            ni_img.SetSpacing(args.spacing[::-1])
            print(ni_img.GetSpacing(), ni_img.GetSize())
            ni_img_name = patient + "_" + session + ".nii.gz" if session else patient + ".nii.gz"
            sitk.WriteImage(ni_img, os.path.join(output_directory, ni_img_name))

        print("=============")
        print("Overall Mean Dice: {:.3f}".format(np.mean(dice_list_case)))
        dice_score = np.mean(np.array(dice_list_case), axis=0)
        for val_channel_ind in range(len(args.semantic_classes)):
            print("{}_dice: {:.3f}".format(args.semantic_classes[val_channel_ind], dice_score[val_channel_ind]))

        print("Overall Hausdorff distance: {:.3f}".format(np.mean(hd_list_case)))
        dh_score = np.mean(np.array(hd_list_case), axis=0)
        for val_channel_ind in range(len(args.semantic_classes)):
            print("{}_hd: {:.3f}".format(args.semantic_classes[val_channel_ind], dh_score[val_channel_ind]))

        print("Overall Average Surface Distance: {:.3f}".format(np.mean(sd_list_case)))
        sd_score = np.mean(np.array(sd_list_case), axis=0)
        for val_channel_ind in range(len(args.semantic_classes)):
            print("{}_sd: {:.3f}".format(args.semantic_classes[val_channel_ind], sd_score[val_channel_ind]))

        print('Average inference time: %.3f s' % (np.mean(infer_time)))
        print('Average throughput: %.3f volumes/second' % (1 / np.mean(infer_time)))

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

