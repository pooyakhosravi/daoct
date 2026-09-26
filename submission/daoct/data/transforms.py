"""Domain randomization (vendor-invariance) + weak/strong augment for mean-teacher.

Domain randomization is the cheapest, most robust domain-generalization lever and
needs no target data, so it also helps the unseen Triton. We randomize the parts of
the image that differ across OCT vendors: global intensity / gamma / contrast,
speckle noise, blur (PSF), and mild row/col scaling — while leaving layer geometry
intact. All ops act on a torch tensor [B,1,H,W] in [0,1].
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _rand(a, b, n, device):
    return a + (b - a) * torch.rand(n, 1, 1, 1, device=device)


def domain_randomize(x, strength=1.0):
    B = x.shape[0]; dev = x.device
    # gamma
    g = _rand(0.6, 1.6, B, dev) ** (1.0 if strength >= 1 else strength)
    x = x.clamp(1e-4, 1).pow(g)
    # contrast / brightness
    c = _rand(0.7, 1.3, B, dev); m = x.mean(dim=(2, 3), keepdim=True)
    x = (x - m) * c + m + _rand(-0.1, 0.1, B, dev) * strength
    x = x.clamp(0, 1)
    # speckle (multiplicative) + gaussian
    if strength > 0:
        x = x * (1 + (_rand(0.0, 0.25, B, dev) * strength) * torch.randn_like(x))
        x = x + (_rand(0.0, 0.04, B, dev) * strength) * torch.randn_like(x)
    # occasional blur (vendor PSF differences) — fixes blur-shift brittleness
    if strength > 0 and torch.rand(()) < 0.3 * strength:
        x = _gauss_blur(x, sigma=float(0.6 + torch.rand(()) * 1.6))
    return x.clamp(0, 1)


def _gauss_blur(x, sigma=1.0, k=5):
    dev = x.device
    ax = torch.arange(k, device=dev) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = (g / g.sum()).to(x.dtype)
    kx = g.view(1, 1, 1, k); ky = g.view(1, 1, k, 1)
    x = F.conv2d(x, kx.expand(x.shape[1], 1, 1, k), padding=(0, k // 2), groups=x.shape[1])
    x = F.conv2d(x, ky.expand(x.shape[1], 1, k, 1), padding=(k // 2, 0), groups=x.shape[1])
    return x


def paired_geom_aug(x, y, vscale=(0.75, 1.3), hscale=(0.9, 1.1), shift=0.05):
    """Paired geometric aug on image+label to simulate cross-vendor axial/lateral
    sampling differences (the real domain gap: Spectralis ~496 vs Cirrus ~1024 rows).
    x:[B,1,H,W] float, y:[B,1,H,W] long. Same output size; content rescaled."""
    import torch.nn.functional as F
    B = x.shape[0]; dev = x.device
    sv = _rand(*vscale, B, dev).view(B)
    sh = _rand(*hscale, B, dev).view(B)
    ty = (_rand(-shift, shift, B, dev)).view(B)
    theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0] = 1.0 / sh
    theta[:, 1, 1] = 1.0 / sv
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    ya = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=False)
    return xa, ya.long()


def axial_fov_exit(
    x,
    y,
    prob=0.12,
    min_shift=0.45,
    max_shift=1.05,
    top_prob=0.8,
    max_tilt=0.10,
):
    """Move most or all retinal interfaces outside the axial field of view."""
    batch, _, height, width = x.shape
    device = x.device
    dtype = x.dtype
    apply = (torch.rand(batch, device=device) < prob).to(dtype)
    magnitude = (
        min_shift
        + (max_shift - min_shift) * torch.rand(batch, device=device)
    )
    toward_top = torch.rand(batch, device=device) < top_prob
    sign = torch.where(
        toward_top,
        torch.ones(batch, device=device, dtype=dtype),
        -torch.ones(batch, device=device, dtype=dtype),
    )
    offset = apply * sign * magnitude
    tilt = (
        apply
        * (2.0 * torch.rand(batch, device=device, dtype=dtype) - 1.0)
        * max_tilt
    )

    rows = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    columns = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    grid = torch.stack(
        (
            grid_x.expand(batch, height, width),
            grid_y.expand(batch, height, width),
        ),
        dim=-1,
    ).clone()
    grid[..., 1] += offset[:, None, None]
    grid[..., 1] += tilt[:, None, None] * grid_x[None]
    image = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    target = F.grid_sample(
        y.float(),
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=True,
    )
    return image, target.long()


def curvature_warp(x, y, max_amp=0.5, prob=0.6):
    """Bend a (flat-ish macula) B-scan into a widefield-like U/V curve, image+label
    together. Per column we add a smooth parabolic vertical shift (random amplitude,
    sign and centre). This synthesizes the strong retinal curvature of wide-FOV scans
    from the macula labels we have — the dominant WideField failure mode. [B,1,H,W]."""
    import torch.nn.functional as F
    B, _, H, W = x.shape
    dev = x.device
    xs = torch.linspace(-1, 1, W, device=dev)
    ys = torch.linspace(-1, 1, H, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx.expand(B, H, W), gy.expand(B, H, W)], dim=-1).clone()  # [B,H,W,2]
    amp = (torch.rand(B, device=dev) * 2 - 1) * max_amp              # U or inverted-U
    center = torch.rand(B, device=dev) * 1.2 - 0.6
    shape = (xs[None, :] - center[:, None]) ** 2                     # [B,W] parabola
    shape = shape / (shape.max(dim=1, keepdim=True).values + 1e-6)
    apply = (torch.rand(B, device=dev) < prob).float()[:, None]
    dy = (amp[:, None] * shape) * apply                             # [B,W] vertical shift
    grid[..., 1] = grid[..., 1] + dy[:, None, :]                     # broadcast over rows
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    ya = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=True)
    return xa, ya.long()


def rotate_aug(x, y, max_deg=15.0, prob=0.5):
    """Random small rotation of image+label together: simulates tilted / off-axis
    B-scans where the retina runs diagonally (a Dice-chaos failure mode on wide and
    Spectralis 20x20 scans). Mild (<=15 deg) so columns stay ~monotonic for DP-refine.
    x:[B,1,H,W] float, y:[B,1,H,W] long."""
    import math
    import torch.nn.functional as F
    B = x.shape[0]; dev = x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * (max_deg * math.pi / 180.0)
    ang = ang * (torch.rand(B, device=dev) < prob).to(ang.dtype)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    ya = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=False)
    return xa, ya.long()


def crop_zoom_aug(x, y, min_frac=0.5, max_frac=0.85, prob=0.4):
    """Random sub-window crop upsampled back to full size (image+label). Gives the
    network higher effective resolution on a retinal patch so thin layers (RNFL/ELM)
    occupy more pixels -> sharper boundaries (MASD). The training-side form of patch
    detail (no costly tiled inference). x:[B,1,H,W] float, y:[B,1,H,W] long."""
    import torch.nn.functional as F
    B = x.shape[0]; dev = x.device
    f = _rand(min_frac, max_frac, B, dev).view(B)                 # zoom box size (frac of full)
    apply = (torch.rand(B, device=dev) < prob).to(x.dtype)
    s = f * apply + 1.0 * (1 - apply)                             # scale<1 -> zoom in; 1 -> identity
    cx = (torch.rand(B, device=dev) * 2 - 1) * (1 - f) * apply    # box centre, kept inside
    cy = (torch.rand(B, device=dev) * 2 - 1) * (1 - f) * apply
    theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0] = s; theta[:, 1, 1] = s
    theta[:, 0, 2] = cx; theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    ya = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=False)
    return xa, ya.long()


def pathology_warp(x, y, prob=0.3, max_amp=0.14, min_w=0.04, max_w=0.16, max_n=2):
    """Local geometric deformation simulating focal pathology (drusen = outer-retina
    elevation; serous PED / fluid = focal layer push). A narrow Gaussian-in-x vertical
    displacement applied to image+label TOGETHER, biased toward the outer (lower) layers
    so the RPE/photoreceptor band domes up under a flatter inner retina — the real drusen
    geometry. Complements intensity-only lesion_aug (which teaches 'don't panic' but not
    'bend around the lesion'). Paired so the GT stays valid. [B,1,H,W]."""
    import torch.nn.functional as F
    B, _, H, W = x.shape; dev = x.device
    xs = torch.linspace(-1, 1, W, device=dev)
    ys = torch.linspace(-1, 1, H, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx.expand(B, H, W), gy.expand(B, H, W)], dim=-1).clone()
    depth = gy.clamp(min=0.0)[None]                          # 0 top -> 1 bottom: bias outer layers
    dy = torch.zeros(B, H, W, device=dev)
    for _ in range(max_n):
        apply = (torch.rand(B, device=dev) < prob).float()
        center = torch.rand(B, device=dev) * 1.6 - 0.8
        width = min_w + torch.rand(B, device=dev) * (max_w - min_w)
        sign = torch.where(torch.rand(B, device=dev) < 0.8, -1.0, 1.0)   # mostly elevation (up)
        amp = torch.rand(B, device=dev) * max_amp * apply * sign
        bump = torch.exp(-((xs[None, :] - center[:, None]) ** 2) / (2 * width[:, None] ** 2))  # [B,W]
        dy = dy + amp[:, None, None] * bump[:, None, :] * depth          # outer-weighted vertical shift
    grid[..., 1] = grid[..., 1] + dy
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    ya = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=True)
    return xa, ya.long()


def _ordered_surfaces(label, n_bound=9):
    """Return the row of each ordered interface in every A-scan column."""
    lab = label[:, 0]
    return torch.stack(
        [(lab <= index).sum(dim=1).float() for index in range(n_bound)],
        dim=1,
    )


def _warp_between_surfaces(image, old, new):
    """Piecewise-linearly move image rows from old interfaces to new ones."""
    import torch.nn.functional as F

    batch, _, height, width = image.shape
    zero = old.new_zeros((batch, 1, width))
    bottom = old.new_full((batch, 1, width), float(height - 1))
    old_control = torch.cat((zero, old, bottom), dim=1)
    new_control = torch.cat((zero, new, bottom), dim=1)
    rows = torch.arange(
        height, device=image.device, dtype=old.dtype
    ).view(1, height, 1).expand(batch, -1, width)
    interval = (
        rows.unsqueeze(1) >= new.unsqueeze(2)
    ).sum(dim=1).clamp_max(9)

    def gather(control, index):
        expanded = control.unsqueeze(2).expand(-1, -1, height, -1)
        return expanded.gather(1, index.unsqueeze(1)).squeeze(1)

    new_low = gather(new_control, interval)
    new_high = gather(new_control, interval + 1)
    old_low = gather(old_control, interval)
    old_high = gather(old_control, interval + 1)
    fraction = (rows - new_low) / (new_high - new_low).clamp_min(1e-3)
    source_y = old_low + fraction * (old_high - old_low)
    grid_y = 2.0 * source_y / max(height - 1, 1) - 1.0
    grid_x = torch.linspace(
        -1, 1, width, device=image.device, dtype=image.dtype
    ).view(1, 1, width).expand(batch, height, -1)
    grid = torch.stack((grid_x, grid_y.to(image.dtype)), dim=-1)
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border",
        align_corners=True,
    )


def anatomy_warp(x, y, prob=0.35, modes="edema,drusen,onh", severity=0.8):
    """Paired implicit disease/ONH augmentation using only source layer labels.

    Edema expands a middle retinal compartment, drusen elevates outer interfaces,
    and ONH selectively collapses inner interfaces while retaining the outer
    boundary. No disease annotation or external segmentation is required.
    """
    batch, _, height, width = x.shape
    device = x.device
    names = tuple(name.strip() for name in modes.split(",") if name.strip())
    if not names:
        return x, y

    old = _ordered_surfaces(y)
    new = old.clone()
    selected = torch.rand(batch, device=device) < prob
    kind = torch.full((batch,), -1, device=device, dtype=torch.long)
    host = torch.full((batch,), 5, device=device, dtype=torch.long)
    columns = torch.linspace(-1, 1, width, device=device)
    profiles = x.new_zeros((batch, width))

    for index in torch.where(selected)[0].tolist():
        name_index = int(torch.randint(len(names), (), device=device))
        name = names[name_index]
        center = float(torch.empty((), device=device).uniform_(-0.65, 0.65))
        if name in ("onh", "onh_continuous"):
            sigma = float(torch.empty((), device=device).uniform_(0.10, 0.22))
        else:
            sigma = float(torch.empty((), device=device).uniform_(0.04, 0.14))
        profile = torch.exp(-0.5 * ((columns - center) / sigma).square())
        profiles[index] = profile

        if name == "edema":
            kind[index] = 0
            host_index = int(torch.randint(4, 7, (), device=device))
            host[index] = host_index
            weights = old.new_zeros(9)
            weights[:host_index] = torch.linspace(
                -0.45, -0.10, host_index, device=device
            )
            weights[host_index:] = torch.linspace(
                0.35, 0.0, 9 - host_index, device=device
            )
            amplitude = float(
                torch.empty((), device=device).uniform_(0.018, 0.060)
            ) * height * severity
            new[index] += amplitude * weights[:, None] * profile[None]
        elif name == "drusen":
            kind[index] = 1
            weights = old.new_tensor(
                (-0.08, -0.12, -0.18, -0.25, -0.36,
                 -0.52, -0.72, -0.95, 0.0)
            )
            amplitude = float(
                torch.empty((), device=device).uniform_(0.018, 0.055)
            ) * height * severity
            new[index] += (
                amplitude * weights[:, None] * profile.square()[None]
            )
        elif name == "onh":
            kind[index] = 2
            top = old[index, :1]
            spacing = old.new_tensor(
                (0.0, 1.0, 2.0, 3.5, 6.0, 10.0, 16.0, 24.0, 0.0)
            ).view(9, 1)
            collapsed = top + spacing
            collapsed[8] = old[index, 8]
            weights = old.new_tensor(
                (0.0, 0.98, 0.98, 0.92, 0.74, 0.38, 0.16, 0.04, 0.0)
            ).view(9, 1)
            blend = (
                severity * weights * profile.view(1, width)
            ).clamp(0, 0.98)
            new[index] = old[index] * (1 - blend) + collapsed * blend
        elif name == "onh_continuous":
            # Released scan 229 keeps every layer through its raised region.
            kind[index] = 2
            side = -1.0 if bool(torch.rand((), device=device) < 0.5) else 1.0
            center = side * float(
                torch.empty((), device=device).uniform_(0.48, 0.82)
            )
            sigma = float(
                torch.empty((), device=device).uniform_(0.12, 0.25)
            )
            profile = torch.exp(
                -0.5 * ((columns - center) / sigma).square()
            )
            profiles[index] = profile
            weights = old.new_tensor(
                (-1.00, -0.92, -0.84, -0.75, -0.66,
                 -0.56, -0.45, -0.33, -0.20)
            )
            amplitude = float(
                torch.empty((), device=device).uniform_(0.045, 0.12)
            ) * height * severity
            proposed = (
                old[index]
                + amplitude * weights[:, None] * profile[None]
            )
            # Translate the stack before clipping so thin classes survive.
            valid = (old[index, 8] - old[index, 0]) > 4
            if not bool(valid.any()):
                kind[index] = -1
                profiles[index].zero_()
                new[index] = old[index]
                continue
            valid_top = proposed[0, valid]
            valid_bottom = proposed[8, valid]
            shift_down = (2.0 - valid_top.amin()).clamp_min(0.0)
            proposed = proposed + shift_down
            shift_up = (
                (valid_bottom + shift_down).amax() - (height - 3.0)
            ).clamp_min(0.0)
            new[index] = proposed - shift_up
        else:
            raise ValueError(f"unknown anatomy augmentation mode: {name}")

    valid_column = (old[:, 8] - old[:, 0]) > 4
    new = new.clamp(2, height - 3)
    new = torch.cummax(new, dim=1).values
    new = torch.where(valid_column[:, None], new, old)
    warped = _warp_between_surfaces(x, old, new)
    rows = torch.arange(
        height, device=device, dtype=new.dtype
    ).view(1, height, 1)
    target = (
        rows.unsqueeze(1) >= new.unsqueeze(2)
    ).sum(dim=1).long()

    retinal_top = new[:, 0]
    retinal_bottom = new[:, 8]
    band = (
        (rows >= retinal_top[:, None])
        & (rows <= retinal_bottom[:, None])
    ).float()
    generic_mid = 0.5 * (new[:, 3] + new[:, 6])
    generic_radius = (0.05 * height) + 0.15 * (
        new[:, 6] - new[:, 3]
    ).clamp_min(2.0)
    vertical = torch.exp(
        -2.0 * ((rows - generic_mid[:, None]) / generic_radius[:, None]).square()
    )
    lesion = profiles[:, None] * vertical
    retinal_profile = profiles[:, None] * band

    fluid = (kind == 0)[:, None, None, None]
    drusen = (kind == 1)[:, None, None, None]
    onh = (kind == 2)[:, None, None, None]
    warped = torch.where(
        fluid, warped * (1.0 - 0.62 * severity * lesion[:, None]), warped
    )
    warped = torch.where(
        drusen, warped + 0.14 * severity * lesion[:, None], warped
    )
    warped = torch.where(
        onh, warped + 0.08 * severity * retinal_profile[:, None], warped
    )
    below = (
        rows >= new[:, 8][:, None]
    ).float() * profiles[:, None]
    warped = torch.where(
        onh, warped * (1.0 - 0.18 * severity * below[:, None]), warped
    )
    warped = warped.masked_fill(~valid_column[:, None, None, :], 0.0)
    target = target.masked_fill(~valid_column[:, None], 0)
    return warped.clamp(0, 1), target[:, None]


def lesion_aug(x, prob=0.3, max_n=2):
    """Inject synthetic low-contrast elliptical intensity blobs (hypo=fluid pocket,
    hyper=drusen/exudate) WITHOUT touching the label: teaches the model to keep
    tracking layers through unfamiliar pathology-like signals instead of producing
    chaotic interiors (the red/cyan-blob failures on Spectralis 20x20). image-only."""
    B, _, H, W = x.shape; dev = x.device
    ys = torch.linspace(0, 1, H, device=dev).view(H, 1)
    xs = torch.linspace(0, 1, W, device=dev).view(1, W)
    out = x.clone()
    for b in range(B):
        if torch.rand(()) >= prob:
            continue
        for _ in range(int(torch.randint(1, max_n + 1, (1,)))):
            cy = float(torch.rand(())); cx = float(torch.rand(()))
            ry = float(0.03 + torch.rand(()) * 0.12); rx = float(0.03 + torch.rand(()) * 0.18)
            blob = torch.exp(-(((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2))   # soft ellipse [H,W]
            amp = float((torch.rand(()) * 2 - 1) * 0.5)
            out[b, 0] = (out[b, 0] + amp * blob).clamp(0, 1)
    return out


def ascan_compress(x, y, prob=0.6, max_comp=2.0, n_classes=10):
    """Per-A-scan vertical COMPRESSION = axial foreshortening at steep/tilted walls.

    Physics: each A-scan (column) is captured at once, so the vertical layer ORDER and
    relationship within a column is reliable; when the retina is steep/tilted relative to
    the A-scan the same layers project onto FEWER axial pixels. So we squeeze each column's
    content vertically around its retinal centroid by a factor k(x)>=1 that is strongest at
    the lateral EDGES (the curve sides) — simulating thin, few-pixel layers there — while
    keeping per-column order (affine-around-centroid is monotonic) and smooth lateral
    variation (B-scans are continuous). Paired image+label. [B,1,H,W]."""
    import torch.nn.functional as F
    B, _, H, W = x.shape; dev = x.device
    apply = (torch.rand(B, device=dev) < prob).float()
    xs = torch.linspace(-1, 1, W, device=dev)
    side = xs.abs()[None, :]                                       # 0 center -> 1 edges
    comp = 1.0 + (max_comp - 1.0) * torch.rand(B, device=dev)      # peak compression per image
    p = (1.0 + 2.5 * torch.rand(B, device=dev))[:, None]          # edge sharpness
    k = 1.0 + (comp[:, None] - 1.0) * (side ** p)                  # [B,W] >=1 (more at edges)
    k = 1.0 + (k - 1.0) * apply[:, None]
    lab = y[:, 0]                                                  # retinal centroid per column
    retina = ((lab >= 1) & (lab <= n_classes - 2)).float()
    rows = torch.arange(H, device=dev, dtype=torch.float32).view(1, H, 1)
    centroid = (retina * rows).sum(1) / retina.sum(1).clamp_min(1.0)   # [B,W] px
    c = (centroid / (H - 1)) * 2 - 1                               # [-1,1]
    ys = torch.linspace(-1, 1, H, device=dev).view(1, H, 1)
    gy = c[:, None, :] + (ys - c[:, None, :]) * k[:, None, :]      # squeeze around centroid
    gx = xs.view(1, 1, W).expand(B, H, W)
    grid = torch.stack([gx, gy], dim=-1)
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    ya = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=True)
    return xa, ya.long()


def steep_edge_warp(x, y, prob=0.4, max_amp=1.3, min_steep=2.5, max_steep=6.0):
    """Push ONE lateral edge into a near-VERTICAL wall — the widefield/peripheral
    geometry where the retina curves steeply up out of frame and the 8 layers compress
    into a few pixels (where the model currently collapses/breaks). curvature_warp makes
    smooth symmetric U-curves; this makes the steep EDGE walls. An exponential vertical
    rise concentrated near the chosen edge. Paired image+label. [B,1,H,W]."""
    import torch.nn.functional as F
    B, _, H, W = x.shape; dev = x.device
    xs01 = torch.linspace(0, 1, W, device=dev)
    ys = torch.linspace(-1, 1, H, device=dev); xs = torch.linspace(-1, 1, W, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx.expand(B, H, W), gy.expand(B, H, W)], dim=-1).clone()
    apply = (torch.rand(B, device=dev) < prob).float()
    left = (torch.rand(B, device=dev) < 0.5)
    steep = min_steep + torch.rand(B, device=dev) * (max_steep - min_steep)
    amp = (torch.rand(B, device=dev) * max_amp) * apply
    edge_d = torch.where(left[:, None], xs01[None, :], 1 - xs01[None, :])   # 0 at chosen edge -> 1
    prof = torch.exp(-steep[:, None] * edge_d)                              # ~1 at edge, steep decay inward
    dy = -(amp[:, None] * prof)                                             # raise retina near the edge
    grid[..., 1] = grid[..., 1] + dy[:, None, :]
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    ya = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=True)
    return xa, ya.long()


def edge_blackout(x, y, max_frac=0.22, bg=0, prob=0.5):
    """Black out random left/right edge columns + set their label to background.
    Teaches the model that no-signal columns (post-registration B-scan shifts, where
    the device pushes the retina to centre and pads edges black) carry NO layers."""
    B, _, H, W = x.shape
    for b in range(B):
        if torch.rand(()) >= prob:
            continue
        wl = int(torch.randint(0, int(max_frac * W) + 1, (1,)))
        wr = int(torch.randint(0, int(max_frac * W) + 1, (1,)))
        if wl:
            x[b, :, :, :wl] = 0; y[b, :, :, :wl] = bg
        if wr:
            x[b, :, :, W - wr:] = 0; y[b, :, :, W - wr:] = bg
    return x, y


def thickness_jitter(x, y, prob=0.5, gmin=0.6, gmax=1.6):
    """Depth-dependent vertical remap (row coord ^ gamma) -> NON-uniform layer-thickness
    change (inner vs outer scaled differently), breaking the constant-thickness prior that
    causes foveal RNFL non-thinning and thick-RNFL splitting. Monotonic -> ordering kept.
    Paired image+label. [B,1,H,W]."""
    import torch.nn.functional as F
    B, _, H, W = x.shape; dev = x.device
    g = _rand(gmin, gmax, B, dev).view(B)
    g = torch.where(torch.rand(B, device=dev) < prob, g, torch.ones_like(g))
    ys = torch.linspace(0, 1, H, device=dev)
    gy = (ys[None, :] ** g[:, None]) * 2 - 1                  # [B,H] monotonic remap -> [-1,1]
    xs = torch.linspace(-1, 1, W, device=dev)
    grid = torch.empty(B, H, W, 2, device=dev)
    grid[..., 0] = xs[None, None, :]
    grid[..., 1] = gy[:, :, None]
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    ya = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=True)
    return xa, ya.long()


def elastic_aug(x, y, prob=0.5, alpha=0.06):
    """Random smooth elastic deformation (low-res gaussian field, upsampled) on image+
    label together. Standard medical-imaging regularizer -> local shape/thickness variation
    and better generalization. [B,1,H,W]."""
    import torch.nn.functional as F
    if torch.rand(()) >= prob:
        return x, y
    B, _, H, W = x.shape; dev = x.device
    ch, cw = max(2, H // 32), max(2, W // 32)
    d = torch.randn(B, 2, ch, cw, device=dev) * alpha
    d = F.interpolate(d, size=(H, W), mode="bicubic", align_corners=True)
    ys = torch.linspace(-1, 1, H, device=dev); xs = torch.linspace(-1, 1, W, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx.expand(B, H, W) + d[:, 0], gy.expand(B, H, W) + d[:, 1]], dim=-1)
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    ya = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=True)
    return xa, ya.long()


def device_intensity_aug(x, prob=0.7):
    """Simulate cross-device intensity character BEYOND a global stretch, so the model
    relies on layer geometry/edges not absolute per-depth brightness:
      - smooth AXIAL (depth) gain profile: SD-OCT signal rolls off with depth while
        swept-source (Triton) penetrates deeper -> different depth brightness;
      - inter-layer CONTRAST jitter (random unsharp/soften): Triton's layer-to-layer
        contrast differs from SD devices.
    Image-only (labels unchanged). [B,1,H,W]."""
    B, _, H, W = x.shape; dev = x.device
    ys = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1)
    lin = _rand(-0.45, 0.45, B, dev)                              # linear depth trend
    bc = torch.rand(B, 1, 1, 1, device=dev)                       # band centre (random layer)
    bw = 0.08 + 0.30 * torch.rand(B, 1, 1, 1, device=dev)
    band = _rand(-0.35, 0.35, B, dev) * torch.exp(-((ys - bc) ** 2) / (2 * bw ** 2))
    prof = (1.0 + lin * ys + band).clamp(0.3, 1.8)               # [B,1,H,1] smooth depth gain
    out = (x * prof).clamp(0, 1)
    if torch.rand(()) < prob:                                     # inter-layer contrast jitter
        blur = _gauss_blur(out, sigma=float(0.8 + torch.rand(()) * 1.6))
        amt = float(torch.rand(()) * 1.4 - 0.3)                  # >0 sharpen, <0 soften
        out = (out + amt * (out - blur)).clamp(0, 1)
    return out


def noise_transfer(x, tgt, prob=0.4, max_strength=1.2):
    """SVDNA-inspired device-NOISE transfer [Koch 2022]: add an unlabeled target B-scan's
    high-frequency speckle residual onto the (structure-preserving) source image. Cheap
    GPU proxy for full SVD noise-transfer — targets the swept-source (Triton) vs spectral-
    domain speckle/texture gap that FDA (low-freq style) doesn't cover. image-only."""
    if torch.rand(()) >= prob:
        return x
    s = float(1.0 + torch.rand(()) * 1.2)
    t_hf = tgt - _gauss_blur(tgt, sigma=s)                # target speckle/texture residual
    amt = float(0.4 + torch.rand(()) * (max_strength - 0.4))
    return (x + amt * t_hf[: x.shape[0]]).clamp(0, 1)


def weak_aug(x):
    return domain_randomize(x, strength=0.3)


def strong_aug(x):
    x = domain_randomize(x, strength=1.0)
    if torch.rand(()) < 0.5:
        x = _gauss_blur(x, sigma=float(0.5 + torch.rand(()) * 1.5))
    return x.clamp(0, 1)
