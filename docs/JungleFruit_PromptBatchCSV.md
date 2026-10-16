# Prompt Batch from CSV

Drive a whole batch from a spreadsheet. Each queued run outputs one row; with the `index`
control set to **increment**, queueing *N* runs renders the full sheet with no manual edits.

## The sheet

| column | required | type | notes |
|---|---|---|---|
| `prompt` | yes | text | must not be empty |
| `negative` | yes | text | may be empty |
| `seed` | yes | integer | 0 to 2^64-1, exact |
| `cfg` | no | number | falls back to `default_cfg` |
| `steps` | no | integer | falls back to `default_steps` |
| `width`, `height` | no | integer | fall back to `default_width` / `default_height` |
| `tag` | no | text | free label (shot name, client code...) |

Headers are case-insensitive, column order is free, comma / semicolon / tab are detected,
a UTF-8 BOM (Excel export) is fine and blank lines are skipped. Errors name the line and
column, e.g. `line 4: column 'seed' must be an integer, got 'abc'`.

## Inputs

| input | type | default | |
|---|---|---|---|
| `csv_path` | STRING | `prompts.csv` | file inside ComfyUI's `input/` folder; paths that escape it are rejected |
| `csv_text` | STRING (multiline) | empty | paste CSV here instead; wins over `csv_path` |
| `index` | INT + control | 0, **increment** | row to output (0 = first data row) |
| `on_overflow` | COMBO | `wrap` | past the last row: `wrap`, `clamp` or `error` (stop the queue) |
| `default_cfg/steps/width/height` | FLOAT/INT | 7.0 / 20 / 1024 / 1024 | advanced |

## Outputs

`prompt` STRING, `negative` STRING, `seed` INT, `cfg` FLOAT, `steps` INT, `width` INT,
`height` INT, `tag` STRING, `row_index` INT (after wrap/clamp), `total_rows` INT.

The node re-runs when the CSV file changes on disk, not only when a widget changes.
