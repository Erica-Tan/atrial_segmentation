import torch
import torch.nn as nn


from networks.dim3.guided_diffusion.basic_unet import BasicUNetEncoder
from networks.dim3.guided_diffusion.basic_unet_denose import BasicUNetDe
from networks.dim3.guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from networks.dim3.guided_diffusion.respace import SpacedDiffusion, space_timesteps
from networks.dim3.guided_diffusion.resample import UniformSampler


class DiffUNet(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.embed_model = BasicUNetEncoder(3, 1, 2, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, self.in_channels+self.out_channels, self.out_channels, [64, 64, 128, 256, 512, 64],
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            # print(image.shape, x.shape)
            embeddings = self.embed_model(image)
            # print(len(embeddings), embeddings[0].shape)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, self.out_channels, *self.patch_size), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out


if __name__ == "__main__":
    image, label = torch.randn((1, 1, 32, 96, 96)).cuda(0), torch.randn((1, 4, 32, 96, 96)).cuda(0)
    x_start = label

    x_start = (x_start) * 2 - 1

    model = DiffUNet(in_channels=1, out_channels=4, patch_size=(32, 96, 96))

    model.cuda(0)

    print(x_start.shape)
    #
    # x_t, t, noise = model(x=x_start, pred_type="q_sample")
    # print(x_t.shape, t, image.shape)
    #
    # pred_xstart = model(image=image, x=x_t, step=t, pred_type="denoise")
    #
    #
    # print(pred_xstart.shape)
    # # assert preds.shape == x.shape

    # pred_xstart = model(image=image, pred_type="ddim_sample")

    from monai.inferers import SlidingWindowInferer

    image, label = torch.randn((1, 1, 176, 309, 309)).cuda(0), torch.randn((1, 14, 176, 309, 309)).cuda(0)

    slice_infer = SlidingWindowInferer(
        roi_size=(32, 96, 96),
        sw_batch_size=1,
        overlap=0.5,
        progress=True
    )

    from functools import partial

    model_inferer = partial(slice_infer, network=model)

    my_first_dict = {"pred_type": "ddim_sample"}
    pred = model_inferer(image, **my_first_dict)
    print(pred.shape)