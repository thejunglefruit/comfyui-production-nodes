# Apply .cube LUT

Give every frame of a campaign the same look as the edit and the grade. Reads standard 3D
`.cube` files (Resolve, Premiere, After Effects, OCIO bakes) and applies them on the GPU or
CPU with trilinear interpolation (`torch.nn.functional.grid_sample`, one call per image).

## Where LUTs come from

The pack registers a `luts` model folder:

- `ComfyUI/models/luts/` for your own LUTs (create it; sub-folders are fine),
- the pack's `luts/` folder with two example looks, `teal_orange.cube` and
  `bleach_bypass.cube`. They are computed by `scripts/make_assets.py` from formulas, not
  downloaded, so they share the code's licence.

Extra locations can be added under a `luts:` key in `extra_model_paths.yaml`.

## Inputs

| input | type | default | |
|---|---|---|---|
| `image` | IMAGE | | RGB or RGBA (alpha is passed through) |
| `lut_name` | COMBO | first LUT found | every `.cube` in the `luts` folders |
| `intensity` | FLOAT slider | 1.0 | 0 = original, 1 = full look |
| `lut_path` | STRING | empty | advanced override; must resolve inside a `luts` folder or `input/` |

## Output

`image` IMAGE, same shape as the input.

## Supported .cube features

`TITLE`, `LUT_3D_SIZE` (2-256), `DOMAIN_MIN`, `DOMAIN_MAX`, `LUT_3D_INPUT_RANGE`, comments.
1D LUTs are rejected with a clear message. Parsed LUTs are cached and reloaded if the file
changes. The LUT is applied to the values as they are (ComfyUI images are display-referred
sRGB), so use LUTs designed for display-referred input, e.g. Rec.709 looks.
