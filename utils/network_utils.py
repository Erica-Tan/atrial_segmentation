def get_model(args):
    if args.model_name == 'resunet':
        from monai.networks.nets import UNet
        return UNet(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            kernel_size=5,
            up_kernel_size=5,
            norm=args.norm_name,
            act=args.act,
            dropout = args.dropout_rate,
        )
    elif args.model_name == 'unetplusplus':
        from networks.dim2.basic_unetplusplus import BasicUNetPlusPlus
        return BasicUNetPlusPlus(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            features=(32, 64, 128, 256, 512, 32),
            deep_supervision=args.deep_supervision,
            kernel_size=5,
            norm=args.norm_name,
            act=args.act,
            dropout=args.dropout_rate,
        )
    elif args.model_name == 'attentionunet':
        from networks.dim2.attention_unet import AttentionUnet
        return AttentionUnet(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            kernel_size=5,
            up_kernel_size=5,
            norm=args.norm_name,
            act=args.act,
            dropout=args.dropout_rate,
        )
    elif args.model_name == 'vnet':
        from monai.networks.nets import VNet
        return VNet(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            act=args.act,
            dropout_prob=args.dropout_rate,
        )

    if args.spatial_dims == 3:
        if args.model_name == 'dynunet':
            from monai.networks.nets import DynUNet

            kernel_size = [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
            strides = [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
            num_stages = len(kernel_size)
            UNet_base_num_features = 32
            unet_max_num_features = 320
            features_per_stage = [min(UNet_base_num_features * 2 ** i,
                                      unet_max_num_features) for i in range(num_stages)]

            return DynUNet(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                kernel_size=kernel_size,
                strides=strides,
                filters=features_per_stage,
                upsample_kernel_size=strides[1:],
                norm_name=args.norm_name,
                deep_supervision=False,
                # deep_supr_num=3,
                res_block=2
            )
        elif args.model_name == 'unetr':
            from networks.dim3.unetr import UNETR
            return UNETR(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
            )
        elif args.model_name == 'swinunetr':
            from monai.networks.nets import SwinUNETR
            return SwinUNETR(
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                feature_size=args.feature_size,
                drop_rate=args.dropout_rate,
                attn_drop_rate=0.0,
                dropout_path_rate=args.dropout_path_rate,
                use_checkpoint=True,
            )
        elif args.model_name == 'diffunet':
            from networks.dim3.diff_unet import DiffUNet
            return DiffUNet(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                patch_size=(args.roi_x, args.roi_y, args.roi_z)
            )

    raise ValueError("Unsupported model " + str(args.model_name))



if __name__ == "__main__":
    import torch
    from ptflops import get_model_complexity_info
    from networks.dim2.basic_unetplusplus import BasicUNetPlusPlus
    
    model = BasicUNetPlusPlus(
             spatial_dims=2,
             in_channels=1,
             out_channels=3,
             features=(32, 64, 128, 256, 512, 32),
             norm='instance',
             dropout=0.2,
             deep_supervision=True,
             act='prelu',
             kernel_size=5
         )

    # print(model)

    # x = torch.randn((1, 1, 320, 320))
    # print(x.shape, x.dtype)
    # output = model(x)
    # # print(output[0].shape)

