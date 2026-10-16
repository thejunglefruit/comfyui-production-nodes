# Quality Gate

Automatic first-pass QC for generated batches. It loads no model: every metric is a few
tensor operations on Rec.709 luminance, so it adds milliseconds, not VRAM.

| metric | how | catches |
|---|---|---|
| sharpness | variance of the 4-neighbour Laplacian, 0-255 scale | mushy, out-of-focus or collapsed renders |
| mean brightness | mean luminance, 0-1 | black or washed-out frames |
| clipped highlights | % of pixels at >= 98% luminance | blown skies, white blowouts |
| crushed shadows | % of pixels at <= 2% luminance | blocked-up blacks |
| contrast | RMS contrast (std of luminance) | flat, foggy frames |

## Inputs

| input | type | default |
|---|---|---|
| `images` | IMAGE | |
| `min_sharpness` | FLOAT | 50 |
| `min_brightness` / `max_brightness` | FLOAT | 0.08 / 0.92 |
| `max_clipped_highlights` | FLOAT | 0.05 |
| `max_crushed_shadows` | FLOAT | 0.05 |
| `min_contrast` | FLOAT | 0.04 |
| `if_none_pass` | COMBO | `return_best` |

Sharpness depends on resolution and subject (a misty landscape is legitimately soft), so
calibrate `min_sharpness` once per project: run a batch, read the report, set the bar.

## Outputs

| output | type | |
|---|---|---|
| `passed` | IMAGE | images that met every threshold |
| `rejected` | IMAGE | the others |
| `report` | STRING | JSON: thresholds, counts, metrics and reasons per image |
| `pass_count` | INT | number of images that genuinely passed |

## Empty branches (design decision)

An empty IMAGE batch crashes many downstream nodes, so the gate never emits one:

- **Nothing rejected**: the `rejected` output is blocked silently, so whatever is connected
  to it (a preview, a "rejects" save) just does not run. No error is raised.
- **Nothing passed**:
  - `return_best` (default): `passed` carries the single best image (fewest failed checks,
    then sharpest). The report sets `"fallback_used": true` and `pass_count` stays 0, so a
    pipeline can still tell that nothing really passed.
  - `block`: `passed` is blocked silently and its downstream branch is skipped for this run.

Example report entry:

```json
{"index": 2, "passed": false, "sharpness": 12.4, "mean_luma": 0.97,
 "clipped_highlights": 0.81, "crushed_shadows": 0.0, "contrast": 0.05,
 "reasons": ["soft: sharpness 12.4 < 50", "overexposed: mean luma 0.970 > 0.92",
             "clipped highlights: 81.0% > 5.0%"]}
```
