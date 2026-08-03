import numpy as np
import torch


def _normalize_xy_for_prompt(coord: list | np.ndarray | torch.Tensor, xy_coord_range: str) -> list:
    if isinstance(coord, torch.Tensor):
        coord = coord.detach().cpu().numpy()
    c = np.asarray(coord, dtype=np.float32).copy()
    if xy_coord_range == "neg1_1":
        c[0] = (c[0] + 1.0) / 2.0
        c[1] = (c[1] + 1.0) / 2.0
    elif xy_coord_range == "zero_1":
        pass
    else:
        raise ValueError(f"Unsupported xy_coord_range: {xy_coord_range}")
    return c.tolist()


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


def get_edit_prompt(
    object_name: list[str],
    coordinates: list[list | np.ndarray | torch.Tensor],
    object_edit_prompt: list[str | None] | None = None,
    additional_prompt: str | None = None,
    xy_coord_range: str = "neg1_1",
) -> str:
    instruction_list = []
    for i in range(len(object_name)):
        src_coord_ = coordinates[i][0]
        tgt_coord_ = coordinates[i][1]

        if object_edit_prompt is None:
            object_edit_prompt_ = ""
        else:
            if object_edit_prompt[i] is None or "no change" in object_edit_prompt[i].strip().lower():
                object_edit_prompt_ = ""
            else:
                object_edit_prompt_ = object_edit_prompt[i].strip().rstrip(".")

        src_coord_ = _normalize_xy_for_prompt(src_coord_, xy_coord_range=xy_coord_range)
        tgt_coord_ = _normalize_xy_for_prompt(tgt_coord_, xy_coord_range=xy_coord_range)

        # convert coordinates to string
        src_coord_str = convert_single_coordinate_to_str(src_coord_)
        tgt_coord_str = convert_single_coordinate_to_str(tgt_coord_)

        edit_prompt_ = f"move the {object_name[i]} from {src_coord_str} to {tgt_coord_str}"
        if object_edit_prompt_:
            edit_prompt_ += f" and {object_edit_prompt_}"
        instruction_list.append(edit_prompt_)

    move_instruction = "Please " + "; ".join(instruction_list) + ". "

    is_multiple = len(object_name) > 1
    subj = "the objects" if is_multiple else "the object"
    possessive = "their" if is_multiple else "its"
    verb = "fit" if is_multiple else "fits"

    # Global edits change which regions should remain untouched.
    if additional_prompt and additional_prompt.strip():
        global_extra_str = f"In addition to these movements, {additional_prompt.strip().rstrip('.')}. "
        constraint = (
            f"Ensure {subj}'s lighting and posture naturally {verb} {possessive} new {('positions' if is_multiple else 'position')}. "
            f"Maintain the identity of {subj} and keep regions not mentioned in the instructions unchanged."
        )

    else:
        global_extra_str = ""
        constraint = (
            f"Ensure {subj}'s lighting and posture naturally {verb} {possessive} new {('positions' if is_multiple else 'position')}, "
            f"while keeping {possessive} identity and features unchanged."
        )

    origin_str = "Assume Picture 1 exists in a 3D space where the origin is at the top-left-near corner. The X-axis ranges from left (0) to right (1), the Y-axis ranges from top (0) to bottom (1), and the Z-axis (depth) ranges from near (0) to far (1). "
    ref_pic_2_str = "Picture 2 is the result applying only the geometric movements described above. Use Picture 2 as a reference for the target positions of the objects. "

    edit_prompt = f"{origin_str}{move_instruction}{ref_pic_2_str}{global_extra_str}{constraint}"
    return edit_prompt


def get_edit_prompt_coord_only(
    object_name: list[str],
    coordinates: list[list | np.ndarray | torch.Tensor],
    object_edit_prompt: list[str | None] | None = None,
    additional_prompt: str | None = None,
    xy_coord_range: str = "neg1_1",
) -> str:
    instruction_list = []
    for i in range(len(object_name)):
        src_coord_ = coordinates[i][0]
        tgt_coord_ = coordinates[i][1]

        if object_edit_prompt is None:
            object_edit_prompt_ = ""
        else:
            if object_edit_prompt[i] is None or "no change" in object_edit_prompt[i].strip().lower():
                object_edit_prompt_ = ""
            else:
                object_edit_prompt_ = object_edit_prompt[i].strip().rstrip(".")

        src_coord_ = _normalize_xy_for_prompt(src_coord_, xy_coord_range=xy_coord_range)
        tgt_coord_ = _normalize_xy_for_prompt(tgt_coord_, xy_coord_range=xy_coord_range)

        # convert coordinates to string
        src_coord_str = convert_single_coordinate_to_str(src_coord_)
        tgt_coord_str = convert_single_coordinate_to_str(tgt_coord_)

        edit_prompt_ = f"move the {object_name[i]} from {src_coord_str} to {tgt_coord_str}"
        if object_edit_prompt_:
            edit_prompt_ += f" and {object_edit_prompt_}"
        instruction_list.append(edit_prompt_)

    move_instruction = "Please " + "; ".join(instruction_list) + ". "

    is_multiple = len(object_name) > 1
    subj = "the objects" if is_multiple else "the object"
    possessive = "their" if is_multiple else "its"
    verb = "fit" if is_multiple else "fits"

    # Global edits change which regions should remain untouched.
    if additional_prompt and additional_prompt.strip():
        global_extra_str = f"In addition to these movements, {additional_prompt.strip().rstrip('.')}. "
        constraint = (
            f"Ensure {subj}'s lighting and posture naturally {verb} {possessive} new {('positions' if is_multiple else 'position')}. "
            f"Maintain the identity of {subj} and keep regions not mentioned in the instructions unchanged."
        )

    else:
        global_extra_str = ""
        constraint = (
            f"Ensure {subj}'s lighting and posture naturally {verb} {possessive} new {('positions' if is_multiple else 'position')}, "
            f"while keeping {possessive} identity and features unchanged."
        )

    origin_str = "Assume the picture exists in a 3D space where the origin is at the top-left-near corner. The X-axis ranges from left (0) to right (1), the Y-axis ranges from top (0) to bottom (1), and the Z-axis (depth) ranges from near (0) to far (1). "
    edit_prompt = f"{origin_str}{move_instruction}{global_extra_str}{constraint}"
    return edit_prompt
