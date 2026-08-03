import numpy as np
import torch


def _normalize_xy_for_prompt(
    coordinate: torch.Tensor | np.ndarray | list,
    xy_coord_range: str,
) -> np.ndarray:
    if isinstance(coordinate, torch.Tensor):
        coordinate = coordinate.detach().cpu().numpy()
    normalized = np.asarray(coordinate, dtype=np.float32).copy()
    if xy_coord_range == "neg1_1":
        normalized[0] = (normalized[0] + 1.0) / 2.0
        normalized[1] = (normalized[1] + 1.0) / 2.0
    elif xy_coord_range != "zero_1":
        raise ValueError(f"Unsupported xy_coord_range: {xy_coord_range}")
    return normalized


def convert_single_coordinate_to_str(
    coordinate: torch.Tensor | np.ndarray | list,
    num_decimal: int = 2,
    join_char: str = ", ",
    wrap_char: str = "()",
) -> str:
    if isinstance(coordinate, torch.Tensor):
        coordinate = coordinate.cpu().numpy()
    coordinate = np.round(coordinate, num_decimal)
    coord_strs = [f"{coord:.{num_decimal}f}" for coord in coordinate.tolist()]
    coord_str = wrap_char[0] + join_char.join(coord_strs) + wrap_char[1]
    return coord_str


def number_to_str(number: torch.Tensor | np.ndarray | float | int, num_decimal: int = 2) -> str:
    if isinstance(number, torch.Tensor):
        number = number.cpu().numpy()
    if isinstance(number, np.ndarray):
        number = number.item()
    number_str = f"{number:.{num_decimal}f}"
    return number_str


def get_simple_edit_prompt(
    object_name: list[str],
    coordinates: list[list | np.ndarray | torch.Tensor],
    object_edit_prompt: list[str | None] | None = None,
    additional_prompt: str | None = None,
    xy_coord_range: str = "neg1_1",
) -> str:
    assert len(object_name) == len(coordinates) and (
        object_edit_prompt is None or len(object_name) == len(object_edit_prompt)
    ), (
        f"Length mismatch among object_name, coordinates, depth_range, and prompt. Got {len(object_name)}, {len(coordinates)}, {len(object_edit_prompt) if object_edit_prompt is not None else 'None'}."
    )
    instruction_list = []
    for i in range(len(object_name)):
        src_coord_ = _normalize_xy_for_prompt(coordinates[i][0], xy_coord_range)
        tgt_coord_ = _normalize_xy_for_prompt(coordinates[i][1], xy_coord_range)

        if (
            object_edit_prompt is None
            or object_edit_prompt[i] is None
            or "no change" in object_edit_prompt[i].strip().lower()
        ):
            object_edit_prompt_ = ""
        else:
            object_edit_prompt_ = object_edit_prompt[i].strip().rstrip(".")

        # convert coordinates to string
        src_coord_str = convert_single_coordinate_to_str(src_coord_)
        tgt_coord_str = convert_single_coordinate_to_str(tgt_coord_)

        edit_prompt_ = f"move the {object_name[i]} from {src_coord_str} to {tgt_coord_str}"
        if object_edit_prompt_:
            edit_prompt_ += f" and {object_edit_prompt_}"
        instruction_list.append(edit_prompt_)

    move_instruction = "; ".join(instruction_list) + ". "
    if additional_prompt and additional_prompt.strip():
        global_extra_str = f"Also, {additional_prompt.strip().rstrip('.')}. "

    else:
        global_extra_str = ""

    edit_prompt = (
        f"Assume the image is in a 3D space with origin (0, 0, 0) at the top-left-near corner. "
        f"X ranges from left (0) to right (1), Y ranges from top (0) to bottom (1), "
        f"and Z ranges from near (0) to far (1). "
        f"{move_instruction}"
        f"{global_extra_str}"
    )
    return edit_prompt
