"""Segmentation backbones (MONAI) + multi-head wrapper.

Default SegResNet: strong + light (small-footprint bonus). `make_model(cfg)` returns
the multi-head model (seg + SDM + surface heads, instance-norm trunk) when
cfg['multihead'] is set, else a plain single-head net."""
from __future__ import annotations
from monai.networks.nets import SegResNet, UNet, DynUNet, MedNeXt


def build_model(name="segresnet", in_channels=1, num_classes=10, init_filters=32,
                norm="instance"):
    name = name.lower()
    if name == "segresnet":
        return SegResNet(spatial_dims=2, in_channels=in_channels, out_channels=num_classes,
                         init_filters=init_filters, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                         norm=norm)
    if name == "mednext":
        return MedNeXt(spatial_dims=2, in_channels=in_channels, out_channels=num_classes,
                       init_filters=init_filters, kernel_size=5, blocks_down=(2, 2, 2, 2),
                       blocks_up=(2, 2, 2, 2), norm_type="group")
    if name == "unet":
        return UNet(spatial_dims=2, in_channels=in_channels, out_channels=num_classes,
                    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2)
    if name == "dynunet":
        return DynUNet(spatial_dims=2, in_channels=in_channels, out_channels=num_classes,
                       kernel_size=[3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 2],
                       upsample_kernel_size=[2, 2, 2, 2], res_block=True)
    raise ValueError(f"unknown model {name}")


def make_model(cfg):
    """Unified factory. Returns a multi-head model (dict output) or a plain net (tensor)."""
    if cfg.get("multihead", False):
        from daoct.models.multihead import MultiHeadSeg
        m = MultiHeadSeg(backbone=cfg.get("model", "segresnet"),
                         num_classes=cfg["num_classes"], feat=cfg.get("feat", 32),
                         norm=cfg.get("norm", "instance"),
                         use_sdm=cfg.get("use_sdm", True),
                         use_surface=cfg.get("use_surface", True),
                         init_filters=cfg.get("trunk_filters", 16),
                         use_deform=cfg.get("use_deform", False),
                         use_flatten=cfg.get("use_flatten", False))
        pt = cfg.get("pretrained_trunk")
        if pt:                                   # SSL-pretrained trunk weights (#2)
            import torch
            sd = torch.load(pt, map_location="cpu")
            miss, unexp = m.trunk.load_state_dict(sd, strict=False)
            print(f"[model] loaded SSL trunk {pt} (missing={len(miss)} unexpected={len(unexp)})")
        return m
    return build_model(cfg.get("model", "segresnet"), num_classes=cfg["num_classes"],
                       init_filters=cfg.get("init_filters", 32), norm=cfg.get("norm", "instance"))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
