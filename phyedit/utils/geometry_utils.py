import numpy as np
import torch
import torch.nn.functional as F
from depth_anything_3.utils.geometry import affine_inverse, unproject_depth
from PIL import Image
from torchvision.transforms.functional import to_tensor


def image2tensor(image, device="cuda"):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if isinstance(image, (Image.Image, np.ndarray)):
        img_tensor = to_tensor(image).to(device)
    elif isinstance(image, torch.Tensor):
        img_tensor = image.to(device)
    else:
        raise ValueError("image must be a file path, PIL Image, numpy array, or torch Tensor.")
    return img_tensor


def mask2tensor(mask, device="cuda"):
    if isinstance(mask, str):
        mask = Image.open(mask).convert("L")
    if isinstance(mask, (Image.Image, np.ndarray)):
        mask_tensor = to_tensor(mask).to(device)
    elif isinstance(mask, torch.Tensor):
        mask_tensor = mask.to(device)
    else:
        raise ValueError("mask must be a file path, PIL Image, numpy array, or torch Tensor.")
    mask_tensor = (mask_tensor > 0).to(dtype=torch.bool)
    return mask_tensor


def torch_morphology(mask_bool, mode="dilate", kernel_size=7):
    """Apply binary dilation or erosion with GPU-native max pooling."""
    pad = kernel_size // 2
    # [H, W] -> [1, 1, H, W]
    x = mask_bool.float().unsqueeze(0).unsqueeze(0)
    if mode == "dilate":
        res = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    else:  # erode
        res = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    return res.squeeze() > 0.5


def get_3d_coords(mask_torch, depth_map_torch, orig_shape):
    """Estimate a robust representative 3D anchor with bilinear depth sampling."""
    H, W = orig_shape
    mask_indices = torch.where(mask_torch > 0)
    if len(mask_indices[0]) == 0:
        return None

    y_pix, x_pix = mask_indices[0].float(), mask_indices[1].float()
    grid_x = (x_pix / (W - 1)) * 2 - 1
    grid_y = (y_pix / (H - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(2)

    sampled_z = F.grid_sample(depth_map_torch, grid, mode="bilinear", padding_mode="border", align_corners=True)
    z_vals = sampled_z.reshape(-1)

    z_med = torch.median(z_vals)
    valid_z = z_vals[(z_vals > z_med * 0.8) & (z_vals < z_med * 1.2)]
    repr_z = torch.median(valid_z) if len(valid_z) > 0 else z_med
    repr_x = torch.median(x_pix)
    repr_y = torch.median(y_pix)
    return torch.stack([repr_x, repr_y, repr_z])


def translate_objects_3d_batch(
    images,
    masks,
    target_coords,
    depths,
    intrinsics,
    extrinsics,
    device: torch.device | str = "cuda",
):
    # Materialize each batch input once on the target device.
    images_torch = torch.stack([image2tensor(img, device) for img in images], dim=0)

    masks_list = []
    for m_list in masks:
        if not isinstance(m_list, list):
            m_list = [m_list]
        masks_list.append([mask2tensor(m, device) for m in m_list])

    target_coords_list = []
    for c_list in target_coords:
        if not isinstance(c_list, list):
            c_list = [c_list]
        target_coords_list.append([torch.as_tensor(c, device=device, dtype=torch.float) for c in c_list])

    src_depths_torch = torch.stack(
        [torch.from_numpy(np.load(d) if isinstance(d, str) else d).to(device).float() for d in depths], dim=0
    )

    intrinsics_ts = torch.stack(
        [torch.from_numpy(np.load(i) if isinstance(i, str) else i).to(device).float() for i in intrinsics], dim=0
    )

    extrinsics_ts = torch.stack(
        [torch.from_numpy(np.load(e) if isinstance(e, str) else e).to(device).float() for e in extrinsics], dim=0
    )

    B, _, H_orig, W_orig = images_torch.shape
    H_m, W_m = src_depths_torch.shape[1], src_depths_torch.shape[2]

    # Align depth and intrinsics to the RGB render grid.
    depth_maps = F.interpolate(
        src_depths_torch.unsqueeze(1), size=(H_orig, W_orig), mode="bilinear", align_corners=True
    ).squeeze(1)

    K_orig = intrinsics_ts.clone()
    sw, sh = W_orig / W_m, H_orig / H_m
    K_orig[:, 0, 0] *= sw
    K_orig[:, 1, 1] *= sh
    K_orig[:, 0, 2] *= sw
    K_orig[:, 1, 2] *= sh

    # Unproject the aligned depth map into 3D.
    if extrinsics_ts.shape[1] == 3 and extrinsics_ts.shape[2] == 4:
        extrinsics_homo = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        extrinsics_homo[:, :3, :4] = extrinsics_ts
        extrinsics_ts = extrinsics_homo

    c2w = affine_inverse(extrinsics_ts).unsqueeze(1)
    depth_input = depth_maps.unsqueeze(1).unsqueeze(-1)
    all_points_3d = unproject_depth(depth_input, K_orig.unsqueeze(1), c2w=c2w)

    final_images = images_torch.clone()
    final_depth_maps = depth_maps.clone()
    bg_patch_masks = torch.zeros((B, H_orig, W_orig), dtype=torch.bool, device=device)
    moved_obj_masks_valid = torch.zeros((B, H_orig, W_orig), dtype=torch.bool, device=device)
    moved_obj_masks_all = torch.zeros((B, H_orig, W_orig), dtype=torch.bool, device=device)

    for i in range(B):
        curr_img_masks = masks_list[i]
        curr_img_targets = target_coords_list[i]
        fx_i, fy_i, cx_i, cy_i = K_orig[i, 0, 0], K_orig[i, 1, 1], K_orig[i, 0, 2], K_orig[i, 1, 2]

        # Prepare the source-region patch on the GPU.
        for m in curr_img_masks:
            m_bool = m.squeeze()
            m_dilated = torch_morphology(m_bool, "dilate", 7)
            bg_patch_masks[i] |= m_dilated
            final_images[i, :, m_dilated] = 1.0

            mask_2d = m_dilated.float().view(1, 1, H_orig, W_orig)
            dilated_area = (F.avg_pool2d(mask_2d, 5, stride=1, padding=2) > 0).squeeze()
            edge_bg = dilated_area & (~m_dilated)
            fill_val = depth_maps[i][edge_bg].mean() if edge_bg.any() else torch.median(depth_maps[i][~m_dilated])
            final_depth_maps[i][m_dilated] = fill_val

        # Reproject and render each object independently.
        for m, t_coord in zip(curr_img_masks, curr_img_targets):
            m_bool = m.squeeze()

            # Filter unstable boundary and high-gradient depth pixels.
            m_eroded = torch_morphology(m_bool, "erode", 3)
            dy, dx = torch.gradient(depth_maps[i])
            grad_mask = (torch.sqrt(dx**2 + dy**2)) < (depth_maps[i].std() * 2.5)
            extract_mask = m_eroded & grad_mask

            # Compute the 3D translation from the robust source anchor.
            temp_d_4d = depth_maps[i].view(1, 1, H_orig, W_orig)
            anchor = get_3d_coords(extract_mask, temp_d_4d, (H_orig, W_orig))
            if anchor is None:
                continue

            t_x_pix = (t_coord[0] + 1.0) * (W_orig - 1) / 2.0
            t_y_pix = (t_coord[1] + 1.0) * (H_orig - 1) / 2.0
            t_z = t_coord[2]

            delta_vec = torch.stack(
                [
                    (t_x_pix - cx_i) * t_z / fx_i - (anchor[0] - cx_i) * anchor[2] / fx_i,
                    (t_y_pix - cy_i) * t_z / fy_i - (anchor[1] - cy_i) * anchor[2] / fy_i,
                    t_z - anchor[2],
                ]
            )

            # Forward-project translated object points.
            obj_pts = all_points_3d[i, 0][extract_mask] + delta_vec
            nu = (obj_pts[:, 0] * fx_i / obj_pts[:, 2]) + cx_i
            nv = (obj_pts[:, 1] * fy_i / obj_pts[:, 2]) + cy_i

            valid = (nu >= 0) & (nu < W_orig) & (nv >= 0) & (nv < H_orig) & (obj_pts[:, 2] > 0)
            t_depth = torch.zeros((H_orig, W_orig), device=device)
            t_mask = torch.zeros((H_orig, W_orig), device=device, dtype=torch.bool)

            u_long, v_long = nu[valid].long(), nv[valid].long()
            t_depth[v_long, u_long] = obj_pts[valid, 2]
            t_mask[v_long, u_long] = True

            # Fill projection holes with a GPU morphological close.
            t_mask_filled = torch_morphology(t_mask, "dilate", 7)
            t_mask_filled = torch_morphology(t_mask_filled, "erode", 7)
            moved_obj_masks_all[i] |= t_mask_filled

            view_4d = (1, 1, H_orig, W_orig)
            sum_d = F.avg_pool2d(t_depth.view(view_4d), 5, stride=1, padding=2) * 25
            sum_w = F.avg_pool2d(t_mask.float().view(view_4d), 5, stride=1, padding=2) * 25
            dense_t_depth = torch.where(t_mask, t_depth, (sum_d / (sum_w + 1e-8)).squeeze())

            # Backward-sample RGB values from the source object.
            v_grid, u_grid = torch.where(t_mask_filled)
            z_grid = dense_t_depth[v_grid, u_grid]

            P_orig = (
                torch.stack(
                    [(u_grid.float() - cx_i) * z_grid / fx_i, (v_grid.float() - cy_i) * z_grid / fy_i, z_grid], dim=-1
                )
                - delta_vec
            )

            u_orig_norm = ((P_orig[:, 0] * fx_i / P_orig[:, 2] + cx_i) / (W_orig - 1)) * 2 - 1
            v_orig_norm = ((P_orig[:, 1] * fy_i / P_orig[:, 2] + cy_i) / (H_orig - 1)) * 2 - 1

            # Reject samples outside the source image.
            in_view = (u_orig_norm.abs() <= 1) & (v_orig_norm.abs() <= 1)
            grid = torch.stack([u_orig_norm, v_orig_norm], dim=-1).view(1, 1, -1, 2)

            sampled_rgb = F.grid_sample(images_torch[i].unsqueeze(0), grid, align_corners=True).view(3, -1)
            sampled_m = (
                F.grid_sample(m_bool.float().view(1, 1, H_orig, W_orig), grid, align_corners=True).squeeze() > 0.5
            )

            # Composite with a z-buffer while allowing the erased source patch.
            valid_px = sampled_m & (z_grid > 0) & in_view
            v_f, u_f, z_f = v_grid[valid_px], u_grid[valid_px], z_grid[valid_px]

            # Z-Buffer
            closer = z_f < final_depth_maps[i, v_f, u_f]
            is_in_patch_area = bg_patch_masks[i, v_f, u_f]
            can_render = closer | is_in_patch_area
            v_up, u_up = v_f[can_render], u_f[can_render]

            final_images[i, :, v_up, u_up] = sampled_rgb[:, valid_px][:, can_render]
            final_depth_maps[i, v_up, u_up] = z_f[can_render]
            bg_patch_masks[i, v_up, u_up] = False
            moved_obj_masks_valid[i, v_up, u_up] = True

    return final_images, final_depth_maps, bg_patch_masks, moved_obj_masks_valid, moved_obj_masks_all


def translate_masked_region_3d(
    image,
    mask,
    depth_map,
    intrinsics,
    delta_camera_xyz,
    shift_u_px=0.0,
    shift_v_px=0.0,
    rotation_deg=0.0,
    lock_center_z=True,
    erase_source=True,
    return_render_mask=False,
    respect_input_depth=True,
    extrinsics=None,
    device: torch.device | str = "cuda",
):
    """
    Rigid translation of a masked region in **DA3 / pinhole camera space** (same frame as
    ``unproject_depth`` with identity ``c2w``), then reproject like ``move_objects_3d_batch``.

    If the viewer applies a flip (x, y, z) -> (x, -y, -z), convert UI delta to camera delta with:
    ``delta_camera = (dx_ui, -dy_ui, -dz_ui)``.
    """
    device = torch.device(device)
    delta_vec = torch.as_tensor(delta_camera_xyz, device=device, dtype=torch.float32).reshape(3)

    image_torch = image2tensor(image, device).unsqueeze(0)
    m_bool = mask2tensor(mask, device).squeeze()
    depth_np = np.load(depth_map) if isinstance(depth_map, str) else np.asarray(depth_map)
    depth_t = torch.from_numpy(depth_np).to(device).float().unsqueeze(0)
    K_np = np.load(intrinsics) if isinstance(intrinsics, str) else np.asarray(intrinsics)
    K_base = torch.from_numpy(K_np).to(device).float().unsqueeze(0)

    B, _, H_orig, W_orig = image_torch.shape
    assert B == 1
    H_m, W_m = depth_t.shape[1], depth_t.shape[2]

    depth_maps = F.interpolate(
        depth_t.unsqueeze(1), size=(H_orig, W_orig), mode="bilinear", align_corners=True
    ).squeeze(1)

    K_orig = K_base.clone()
    sw, sh = W_orig / W_m, H_orig / H_m
    K_orig[:, 0, 0] *= sw
    K_orig[:, 1, 1] *= sh
    K_orig[:, 0, 2] *= sw
    K_orig[:, 1, 2] *= sh

    if extrinsics is None:
        extrinsics_ts = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    else:
        extrinsics_ts = torch.from_numpy(np.load(extrinsics) if isinstance(extrinsics, str) else np.asarray(extrinsics))
        extrinsics_ts = extrinsics_ts.to(device).float().unsqueeze(0)
        if extrinsics_ts.shape[1] == 3 and extrinsics_ts.shape[2] == 4:
            extrinsics_homo = torch.eye(4, device=device).unsqueeze(0)
            extrinsics_homo[:, :3, :4] = extrinsics_ts
            extrinsics_ts = extrinsics_homo

    c2w = affine_inverse(extrinsics_ts).unsqueeze(1)
    depth_input = depth_maps.unsqueeze(1).unsqueeze(-1)
    all_points_3d = unproject_depth(depth_input, K_orig.unsqueeze(1), c2w=c2w)

    i = 0
    fx_i, fy_i, cx_i, cy_i = K_orig[i, 0, 0], K_orig[i, 1, 1], K_orig[i, 0, 2], K_orig[i, 1, 2]

    # Keep as many object points as possible for stability when moving closer.
    # Gradient gating can over-prune low-texture regions and make near-camera
    # objects disappear in render while preview (point-cloud projection) is fine.
    extract_mask = m_bool

    if not extract_mask.any():
        empty_mask = torch.zeros((1, H_orig, W_orig), dtype=torch.bool, device=device)
        if return_render_mask:
            return image_torch.clone(), depth_maps.clone(), empty_mask
        return image_torch.clone(), depth_maps.clone()

    final_images = image_torch.clone()
    final_depth_maps = depth_maps.clone()
    bg_patch_masks = torch.zeros((1, H_orig, W_orig), dtype=torch.bool, device=device)
    rendered_mask = torch.zeros((1, H_orig, W_orig), dtype=torch.bool, device=device)

    m_dilated = torch_morphology(m_bool, "dilate", 7)
    if erase_source:
        bg_patch_masks[i] |= m_dilated
        final_images[i, :, m_dilated] = 1.0

        mask_2d = m_dilated.float().view(1, 1, H_orig, W_orig)
        dilated_area = (F.avg_pool2d(mask_2d, 5, stride=1, padding=2) > 0).squeeze()
        edge_bg = dilated_area & (~m_dilated)
        fill_val = depth_maps[i][edge_bg].mean() if edge_bg.any() else torch.median(depth_maps[i][~m_dilated])
        final_depth_maps[i][m_dilated] = fill_val

    obj_pts = all_points_3d[i, 0][extract_mask] + delta_vec
    nu = (obj_pts[:, 0] * fx_i / obj_pts[:, 2]) + cx_i
    nv = (obj_pts[:, 1] * fy_i / obj_pts[:, 2]) + cy_i
    if lock_center_z and abs(float(delta_vec[2].item())) > 1e-8:
        center_u0 = torch.median(torch.where(m_bool)[1].float())
        center_v0 = torch.median(torch.where(m_bool)[0].float())
        z_before = torch.median(all_points_3d[i, 0][extract_mask][:, 2]).clamp(min=1e-6)
        z_after = (z_before + delta_vec[2]).clamp(min=1e-6)
        cu_after = ((center_u0 - cx_i) * z_before / z_after) + cx_i
        cv_after = ((center_v0 - cy_i) * z_before / z_after) + cy_i
        nu = nu + (center_u0 - cu_after)
        nv = nv + (center_v0 - cv_after)
    nu = nu + float(shift_u_px)
    nv = nv + float(shift_v_px)

    # 2D rotation around selected-region center in image plane
    theta = torch.as_tensor(float(rotation_deg), device=device, dtype=torch.float32) * (torch.pi / 180.0)
    has_rot = torch.abs(theta) > 1e-8
    center_u = torch.median(torch.where(m_bool)[1].float()) + float(shift_u_px)
    center_v = torch.median(torch.where(m_bool)[0].float()) + float(shift_v_px)
    if has_rot:
        ct = torch.cos(theta)
        st = torch.sin(theta)
        du = nu - center_u
        dv = nv - center_v
        nu_rot = center_u + du * ct - dv * st
        nv_rot = center_v + du * st + dv * ct
    else:
        nu_rot = nu
        nv_rot = nv

    valid = (nu_rot >= 0) & (nu_rot < W_orig) & (nv_rot >= 0) & (nv_rot < H_orig) & (obj_pts[:, 2] > 0)
    t_depth = torch.zeros((H_orig, W_orig), device=device)
    t_mask = torch.zeros((H_orig, W_orig), device=device, dtype=torch.bool)

    u_long = torch.round(nu_rot[valid]).long()
    v_long = torch.round(nv_rot[valid]).long()
    in_bounds = (u_long >= 0) & (u_long < W_orig) & (v_long >= 0) & (v_long < H_orig)
    u_long = u_long[in_bounds]
    v_long = v_long[in_bounds]
    z_valid = obj_pts[valid, 2][in_bounds]
    t_depth[v_long, u_long] = z_valid
    t_mask[v_long, u_long] = True

    t_mask_filled = torch_morphology(t_mask, "dilate", 7)
    t_mask_filled = torch_morphology(t_mask_filled, "erode", 7)

    view_4d = (1, 1, H_orig, W_orig)
    sum_d = F.avg_pool2d(t_depth.view(view_4d), 5, stride=1, padding=2) * 25
    sum_w = F.avg_pool2d(t_mask.float().view(view_4d), 5, stride=1, padding=2) * 25
    dense_t_depth = torch.where(t_mask, t_depth, (sum_d / (sum_w + 1e-8)).squeeze())

    v_grid, u_grid = torch.where(t_mask_filled)
    z_grid = dense_t_depth[v_grid, u_grid]

    u_back = u_grid.float()
    v_back = v_grid.float()
    if has_rot:
        # inverse-rotate target pixels back to "translated-only" image plane, then backproject
        ct = torch.cos(theta)
        st = torch.sin(theta)
        du = u_back - center_u
        dv = v_back - center_v
        u_back = center_u + du * ct + dv * st
        v_back = center_v - du * st + dv * ct
    # inverse of screen-space shift (forward was +shift_u_px/+shift_v_px)
    u_back = u_back - float(shift_u_px)
    v_back = v_back - float(shift_v_px)

    P_orig = torch.stack([(u_back - cx_i) * z_grid / fx_i, (v_back - cy_i) * z_grid / fy_i, z_grid], dim=-1) - delta_vec

    u_orig_norm = ((P_orig[:, 0] * fx_i / P_orig[:, 2] + cx_i) / (W_orig - 1)) * 2 - 1
    v_orig_norm = ((P_orig[:, 1] * fy_i / P_orig[:, 2] + cy_i) / (H_orig - 1)) * 2 - 1

    in_view = (u_orig_norm.abs() <= 1) & (v_orig_norm.abs() <= 1)
    grid = torch.stack([u_orig_norm, v_orig_norm], dim=-1).view(1, 1, -1, 2)

    sampled_rgb = F.grid_sample(image_torch[i].unsqueeze(0), grid, align_corners=True).view(3, -1)
    m_gate = torch_morphology(m_bool, "dilate", 15).float().view(1, 1, H_orig, W_orig)
    sampled_gate = F.grid_sample(m_gate, grid, align_corners=True).squeeze()
    # Use a soft source-mask gate: strict enough to avoid background color bleeding,
    # but lenient enough to keep near-camera enlarged objects from disappearing.
    valid_px = (sampled_gate > 0.01) & (z_grid > 0) & in_view
    v_f, u_f, z_f = v_grid[valid_px], u_grid[valid_px], z_grid[valid_px]

    if respect_input_depth:
        # A tiny epsilon avoids depth-noise "hard cuts" near equal-depth surfaces.
        closer = z_f <= (final_depth_maps[i, v_f, u_f] + 1e-4)
        can_render = closer
    else:
        # Defer occlusion decision to outer/global compositor.
        can_render = torch.ones_like(z_f, dtype=torch.bool)
    v_up, u_up = v_f[can_render], u_f[can_render]

    final_images[i, :, v_up, u_up] = sampled_rgb[:, valid_px][:, can_render]
    final_depth_maps[i, v_up, u_up] = z_f[can_render]
    bg_patch_masks[i, v_up, u_up] = False
    rendered_mask[i, v_up, u_up] = True

    if return_render_mask:
        return final_images, final_depth_maps, rendered_mask
    return final_images, final_depth_maps
