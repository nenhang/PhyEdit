import torch
from depth_anything_3.utils.geometry import affine_inverse, unproject_depth
from pytorch3d.loss import chamfer_distance
from torch.nn import functional as F


def norm_z_coords(coordinate_lists, reference_depths, gt_depths):
    # Normalize source and target z coordinates over their shared depth range.
    z_normed_coordinate_lists = []
    for coord_list, ref_depth, gt_depth in zip(coordinate_lists, reference_depths, gt_depths):
        min_z = torch.min(torch.cat([ref_depth.flatten(), gt_depth.flatten()])).item()
        max_z = torch.max(torch.cat([ref_depth.flatten(), gt_depth.flatten()])).item()

        single_image_normed_coords = []
        for coord_pair in coord_list:
            src_coord = coord_pair[0]
            tgt_coord = coord_pair[1]
            normed_src_coord = [src_coord[0], src_coord[1], (src_coord[2] - min_z) / (max_z - min_z + 1e-8)]
            normed_tgt_coord = [tgt_coord[0], tgt_coord[1], (tgt_coord[2] - min_z) / (max_z - min_z + 1e-8)]
            single_image_normed_coords.append([normed_src_coord, normed_tgt_coord])
        z_normed_coordinate_lists.append(single_image_normed_coords)
    return z_normed_coordinate_lists


def get_masked_3d_coords(mask_torch, depth_map_torch, orig_shape):
    """Extract a representative 3D anchor using bilinear depth sampling."""
    H, W = orig_shape

    mask_indices = torch.where(mask_torch > 0)

    y_pix, x_pix = mask_indices[0].float(), mask_indices[1].float()

    # grid_sample expects coordinates mapped to [-1, 1].
    # x: [0, W-1] -> [-1, 1], y: [0, H-1] -> [-1, 1]
    grid_x = (x_pix / (W - 1)) * 2 - 1
    grid_y = (y_pix / (H - 1)) * 2 - 1

    # Grid shape: [1, num_pixels, 1, 2].
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(2)

    # Sampled depth shape: [1, 1, num_pixels, 1].
    # align_corners=True keeps the pixel-center mapping consistent.
    sampled_z = F.grid_sample(depth_map_torch, grid, mode="bilinear", padding_mode="border", align_corners=True)

    # Flatten to [num_pixels].
    z_vals = sampled_z.reshape(-1)

    # Median aggregation is robust to mask-boundary outliers.
    repr_z = torch.median(z_vals)
    repr_x = torch.median(x_pix)
    repr_y = torch.median(y_pix)

    return torch.stack([repr_x, repr_y, repr_z])


def norm_coords(coords, image_shape):
    """
    Normalize image-plane coordinates to [0, 1].

    coords: Tensor of shape (N, 3) or (3,)
    image_shape: Tuple (H, W)
    """
    H, W = image_shape
    normed_coords = coords.clone().float()
    normed_coords[..., 0] = normed_coords[..., 0] / W
    normed_coords[..., 1] = normed_coords[..., 1] / H
    # Preserve z because callers normalize depth separately.
    return normed_coords


def cal_coord_2d_depth(coord1, coord2):
    """
    Compute separate image-plane and depth coordinate distances.

    coord1: Tensor of shape (N, 3)
    coord2: Tensor of shape (N, 3)
    """
    assert coord1.shape == coord2.shape, "Shape mismatch between coordinates."

    dist_2d = torch.norm(coord1[:, :2] - coord2[:, :2], dim=1)
    dist_depth = torch.abs(coord1[:, 2] - coord2[:, 2])
    return dist_2d, dist_depth


def cal_bbox_diou(pred_bbox, gt_bbox, eps: float = 1e-8):
    """
    Compute the DIoU score for two 2D bounding boxes.

    Bounding box format: [x1, y1, x2, y2].
    """
    pred = torch.as_tensor(pred_bbox, dtype=torch.float32)
    gt = torch.as_tensor(gt_bbox, dtype=torch.float32, device=pred.device)

    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt

    inter_x1 = torch.maximum(px1, gx1)
    inter_y1 = torch.maximum(py1, gy1)
    inter_x2 = torch.minimum(px2, gx2)
    inter_y2 = torch.minimum(py2, gy2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    pred_area = torch.clamp(px2 - px1, min=0.0) * torch.clamp(py2 - py1, min=0.0)
    gt_area = torch.clamp(gx2 - gx1, min=0.0) * torch.clamp(gy2 - gy1, min=0.0)
    union = pred_area + gt_area - inter_area
    iou = inter_area / (union + eps)

    pred_cx = (px1 + px2) / 2.0
    pred_cy = (py1 + py2) / 2.0
    gt_cx = (gx1 + gx2) / 2.0
    gt_cy = (gy1 + gy2) / 2.0
    center_dist_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

    enc_x1 = torch.minimum(px1, gx1)
    enc_y1 = torch.minimum(py1, gy1)
    enc_x2 = torch.maximum(px2, gx2)
    enc_y2 = torch.maximum(py2, gy2)
    enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

    diou = iou - center_dist_sq / (enc_diag_sq + eps)
    return diou.item()


def cal_masked_mean_depth_distance(
    mask_torch,
    depth_map_torch,
    gt_depth,
    edge_margin: int = 2,
    min_valid_pixels: int = 16,
):
    """
    Compute mean depth error inside a mask.

    Prefer an eroded interior region to reduce unstable boundary pixels.
    """
    if torch.is_tensor(mask_torch):
        mask = mask_torch
    else:
        mask = torch.as_tensor(mask_torch)

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    elif mask.ndim > 2:
        mask = mask.squeeze()

    if depth_map_torch.ndim == 4:
        depth = depth_map_torch[0, 0]
    elif depth_map_torch.ndim == 3:
        depth = depth_map_torch[0]
    else:
        depth = depth_map_torch

    mask = (mask > 0.5).to(depth.device)
    if torch.sum(mask) == 0:
        return float("inf")

    inner_mask = mask
    if edge_margin > 0:
        k = edge_margin * 2 + 1
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
        # Dilating the inverse mask with max pooling erodes the original mask.
        eroded = 1.0 - F.max_pool2d(1.0 - mask_4d, kernel_size=k, stride=1, padding=edge_margin)
        inner_mask = eroded[0, 0] > 0.5

    if torch.sum(inner_mask) < min_valid_pixels:
        inner_mask = mask

    pred_depth_mean = depth[inner_mask].float().mean()
    gt_depth_tensor = torch.as_tensor(float(gt_depth), dtype=pred_depth_mean.dtype, device=pred_depth_mean.device)
    return torch.abs(pred_depth_mean - gt_depth_tensor).item()


def cal_mask_iou(pred_mask, gt_mask, eps: float = 1e-6):
    """Compute IoU between two binary masks."""
    pred = torch.as_tensor(pred_mask).float() > 0.5
    gt = torch.as_tensor(gt_mask).to(device=pred.device).float() > 0.5

    inter = torch.sum(pred & gt).float()
    union = torch.sum(pred | gt).float()
    return (inter / (union + eps)).item()


def cal_depth_absrel_and_delta(
    pred_depth,
    gt_depth,
    eval_mask,
    eps: float = 1e-6,
    delta_thresh: float = 1.25,
):
    """Compute AbsRel and delta < 1.25 inside ``eval_mask``."""
    pred = torch.as_tensor(pred_depth).float()
    gt = torch.as_tensor(gt_depth).to(device=pred.device).float()
    mask = torch.as_tensor(eval_mask).to(device=pred.device)
    mask = mask > 0.5

    valid = mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > eps) & (pred > eps)
    if torch.sum(valid) == 0:
        return float("inf"), 0.0

    p = pred[valid]
    g = gt[valid]

    abs_rel = torch.mean(torch.abs(p - g) / torch.clamp(g, min=eps)).item()
    ratio = torch.maximum(p / torch.clamp(g, min=eps), g / torch.clamp(p, min=eps))
    delta = torch.mean((ratio < delta_thresh).float()).item()
    return abs_rel, delta


def masked_depth_to_points_3d(
    depth_map,
    mask,
    intrinsic,
    extrinsic,
    max_points: int | None = None,
    device=torch.device("cuda"),
):
    """Back-project masked depth pixels into a world-space point cloud."""
    depth = depth_map.to(device=device).float()
    mask_t = mask.to(device=device)
    mask_t = mask_t > 0.5

    if depth.ndim != 2:
        depth = depth.squeeze()
    if mask_t.ndim != 2:
        mask_t = mask_t.squeeze()

    intrinsic_t = intrinsic.to(device=device).float()
    extrinsic_t = extrinsic.to(device=device).float()

    if extrinsic_t.shape == (3, 4):
        extrinsic_h = torch.eye(4, device=device, dtype=extrinsic_t.dtype)
        extrinsic_h[:3, :4] = extrinsic_t
        extrinsic_t = extrinsic_h
    elif extrinsic_t.shape == (3, 3):
        extrinsic_h = torch.eye(4, device=device, dtype=extrinsic_t.dtype)
        extrinsic_h[:3, :3] = extrinsic_t
        extrinsic_t = extrinsic_h
    elif extrinsic_t.shape != (4, 4):
        raise ValueError(f"Unsupported extrinsic shape: {extrinsic_t.shape}")

    H, W = depth.shape
    depth_input = depth.view(1, 1, H, W, 1)
    k_input = intrinsic_t.view(1, 1, 3, 3)
    c2w = affine_inverse(extrinsic_t.view(1, 4, 4)).unsqueeze(1)
    points = unproject_depth(depth_input, k_input, c2w=c2w)[0, 0]  # [H, W, 3]

    flat_points = points[mask_t]
    if flat_points.numel() == 0:
        return flat_points.reshape(0, 3)

    if max_points and flat_points.shape[0] > max_points:
        perm = torch.randperm(flat_points.shape[0], device=device)[:max_points]
        flat_points = flat_points[perm]
    return flat_points


def cal_chamfer_and_centroid_distance(points_a, points_b):
    """Compute point-cloud Chamfer-L2 and centroid distances."""
    pa = torch.as_tensor(points_a).float()
    pb = torch.as_tensor(points_b).to(device=pa.device).float()

    if pa.ndim != 2:
        pa = pa.reshape(-1, 3)
    if pb.ndim != 2:
        pb = pb.reshape(-1, 3)

    if pa.shape[0] == 0 or pb.shape[0] == 0:
        return float("inf"), float("inf")

    chamfer_val = None

    loss_out = chamfer_distance(pa.unsqueeze(0), pb.unsqueeze(0), norm=2)
    if isinstance(loss_out, tuple):
        loss_val = loss_out[0]
    else:
        loss_val = loss_out
    if isinstance(loss_val, tuple):
        loss_val = loss_val[0]
    chamfer_val = float(torch.as_tensor(loss_val).item())

    centroid_dist = torch.norm(pa.mean(dim=0) - pb.mean(dim=0), p=2).item()
    return chamfer_val, centroid_dist


def _subsample_points(points: torch.Tensor, max_points: int | None) -> torch.Tensor:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    # Deterministic subsampling avoids introducing evaluation randomness.
    indices = torch.linspace(0, points.shape[0] - 1, steps=max_points, device=points.device).long()
    return points[indices]


def cal_pointcloud_3d_metrics(
    points_a,
    points_b,
    obj_scale: float,
    threshold_ratios: tuple[float, ...] = (0.01, 0.02, 0.05),
    max_points_for_nn: int = 2048,
    eps: float = 1e-8,
):
    """
    Compute additional 3D point-cloud metrics.

    Returns bidirectional nearest-neighbor L2 mean/P95/Hausdorff distances and
    precision/recall/F-score at thresholds relative to object scale.
    """
    pa = torch.as_tensor(points_a).float()
    pb = torch.as_tensor(points_b).to(device=pa.device).float()

    if pa.ndim != 2:
        pa = pa.reshape(-1, 3)
    if pb.ndim != 2:
        pb = pb.reshape(-1, 3)

    if pa.shape[0] == 0 or pb.shape[0] == 0:
        result = {
            "nn_l2_mean": float("inf"),
            "nn_l2_p95": float("inf"),
            "hausdorff_l2": float("inf"),
        }
        for ratio in threshold_ratios:
            suffix = int(round(ratio * 100))
            result[f"precision_{suffix}pct"] = 0.0
            result[f"recall_{suffix}pct"] = 0.0
            result[f"fscore_{suffix}pct"] = 0.0
        return result

    pa = _subsample_points(pa, max_points_for_nn)
    pb = _subsample_points(pb, max_points_for_nn)

    dmat = torch.cdist(pa, pb, p=2)
    d_a_to_b = torch.min(dmat, dim=1).values
    d_b_to_a = torch.min(dmat, dim=0).values

    nn_l2_mean = 0.5 * (d_a_to_b.mean() + d_b_to_a.mean())
    nn_l2_p95 = 0.5 * (torch.quantile(d_a_to_b, 0.95) + torch.quantile(d_b_to_a, 0.95))
    hausdorff_l2 = torch.maximum(d_a_to_b.max(), d_b_to_a.max())

    scale = max(float(obj_scale), eps)
    result = {
        "nn_l2_mean": float(nn_l2_mean.item()),
        "nn_l2_p95": float(nn_l2_p95.item()),
        "hausdorff_l2": float(hausdorff_l2.item()),
    }

    for ratio in threshold_ratios:
        suffix = int(round(ratio * 100))
        threshold = scale * ratio
        precision = float((d_a_to_b <= threshold).float().mean().item())
        recall = float((d_b_to_a <= threshold).float().mean().item())
        fscore = (2.0 * precision * recall) / (precision + recall + eps)
        result[f"precision_{suffix}pct"] = precision
        result[f"recall_{suffix}pct"] = recall
        result[f"fscore_{suffix}pct"] = float(fscore)

    return result


def cal_motion_projection_penalty(
    points_orig,
    points_pred,
    points_gt,
    eps: float = 1e-8,
    static_motion_ratio: float = 0.02,
    alpha: float = 1.0,
    beta: float = 0.7,
    return_details: bool = False,
):
    """
    Compute the 3D motion projection penalty.

    Definitions:
    - v_pred = centroid(pred) - centroid(orig)
    - v_gt   = centroid(gt)   - centroid(orig)

    The penalty decomposes error along and perpendicular to the target motion:
    - e_parallel: magnitude error along the target direction
    - e_perp: lateral displacement error
    - gate: zero when predicted motion opposes the target direction

    penalty = gate * exp(-alpha * e_parallel - beta * e_perp)

    For near-static targets, the score is 1 when the prediction is also static
    and otherwise decays exponentially with displacement.
    """

    def _pack_result(raw_ratio: float, penalty_val: float, details: dict):
        if return_details:
            return raw_ratio, penalty_val, details
        return raw_ratio, penalty_val

    p_orig = torch.as_tensor(points_orig).float()
    p_pred = torch.as_tensor(points_pred).to(device=p_orig.device).float()
    p_gt = torch.as_tensor(points_gt).to(device=p_orig.device).float()

    if p_orig.ndim != 2:
        p_orig = p_orig.reshape(-1, 3)
    if p_pred.ndim != 2:
        p_pred = p_pred.reshape(-1, 3)
    if p_gt.ndim != 2:
        p_gt = p_gt.reshape(-1, 3)

    if p_orig.shape[0] == 0 or p_pred.shape[0] == 0 or p_gt.shape[0] == 0:
        return _pack_result(
            0.0,
            0.0,
            {
                "motion_proj_formula": "parallel_perp_v1",
                "alpha": float(alpha),
                "beta": float(beta),
                "static_motion_ratio": float(static_motion_ratio),
                "gate": 0.0,
                "err_parallel": 0.0,
                "err_perp": 0.0,
                "dot": 0.0,
                "gt_norm": 0.0,
                "pred_norm": 0.0,
                "is_static_case": True,
                "static_threshold": 0.0,
            },
        )

    c_orig = p_orig.mean(dim=0)
    c_pred = p_pred.mean(dim=0)
    c_gt = p_gt.mean(dim=0)

    v_pred = c_pred - c_orig
    v_gt = c_gt - c_orig

    gt_norm = torch.norm(v_gt, p=2)
    pred_norm = torch.norm(v_pred, p=2)

    ratio_raw = torch.dot(v_pred, v_gt) / (gt_norm * gt_norm + eps)

    gt_center = p_gt.mean(dim=0, keepdim=True)
    gt_scale = torch.norm(p_gt - gt_center, dim=-1).max().clamp(min=1e-6)
    static_eps = gt_scale * static_motion_ratio

    dot_v = torch.dot(v_pred, v_gt)

    if gt_norm <= static_eps:
        if pred_norm <= static_eps:
            return _pack_result(
                float(ratio_raw.item()),
                1.0,
                {
                    "motion_proj_formula": "parallel_perp_v1",
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "static_motion_ratio": float(static_motion_ratio),
                    "gate": 1.0,
                    "err_parallel": 0.0,
                    "err_perp": 0.0,
                    "dot": float(dot_v.item()),
                    "gt_norm": float(gt_norm.item()),
                    "pred_norm": float(pred_norm.item()),
                    "is_static_case": True,
                    "static_threshold": float(static_eps.item()),
                },
            )
        decay = torch.exp(-((pred_norm - static_eps) / (static_eps + eps)))
        return _pack_result(
            float(ratio_raw.item()),
            float(decay.item()),
            {
                "motion_proj_formula": "parallel_perp_v1",
                "alpha": float(alpha),
                "beta": float(beta),
                "static_motion_ratio": float(static_motion_ratio),
                "gate": 0.0,
                "err_parallel": 0.0,
                "err_perp": 0.0,
                "dot": float(dot_v.item()),
                "gt_norm": float(gt_norm.item()),
                "pred_norm": float(pred_norm.item()),
                "is_static_case": True,
                "static_threshold": float(static_eps.item()),
            },
        )

    gate = (dot_v > 0).float()
    v_parallel = (dot_v / (gt_norm * gt_norm + eps)) * v_gt
    v_perp = v_pred - v_parallel
    err_parallel = torch.norm(v_parallel - v_gt, p=2) / (gt_norm + eps)
    err_perp = torch.norm(v_perp, p=2) / (gt_norm + eps)

    penalty = gate * torch.exp(-(alpha * err_parallel + beta * err_perp))
    penalty = torch.clamp(penalty, min=0.0, max=1.0)

    return _pack_result(
        float(ratio_raw.item()),
        float(penalty.item()),
        {
            "motion_proj_formula": "parallel_perp_v1",
            "alpha": float(alpha),
            "beta": float(beta),
            "static_motion_ratio": float(static_motion_ratio),
            "gate": float(gate.item()),
            "err_parallel": float(err_parallel.item()),
            "err_perp": float(err_perp.item()),
            "dot": float(dot_v.item()),
            "gt_norm": float(gt_norm.item()),
            "pred_norm": float(pred_norm.item()),
            "is_static_case": False,
            "static_threshold": float(static_eps.item()),
        },
    )


def cal_masked_coord_distance(
    mask_torch,
    depth_map_torch,
    orig_shape,
    gt_coord,
    device=torch.device("cuda"),
):
    """
    Compute image-plane and depth-coordinate errors inside a mask.

    mask_torch: Binary tensor of shape (H, W)
    depth_map_torch: Depth tensor of shape (1, 1, H, W)
    orig_shape: Original image shape as (H, W)
    gt_coord: Target coordinate tensor of shape (3,)
    """
    pred_coord = get_masked_3d_coords(mask_torch, depth_map_torch, orig_shape).to(device)

    normed_pred_coord = norm_coords(pred_coord, orig_shape)
    # normed_gt_coord = norm_coords(gt_coord.to(device), orig_shape)

    dist_2d, dist_depth = cal_coord_2d_depth(normed_pred_coord.unsqueeze(0), gt_coord.unsqueeze(0))

    return dist_2d.item(), dist_depth.item()
