# Save Images + Run Manifest

A save node that also writes a provenance sidecar next to each PNG, so any delivered frame
can be traced back to the prompt, seed, settings and exact graph that made it, without
opening ComfyUI and without trusting PNG metadata that most tools strip.

```
output/jf/batch_00001_.png
output/jf/batch_00001_.json   <- sidecar
```

## Inputs

| input | type | default | |
|---|---|---|---|
| `images` | IMAGE | | saved as PNG (standard ComfyUI metadata included) |
| `filename_prefix` | STRING | `jf/run` | relative to `output/`; sub-folders allowed |
| `positive`, `negative` | STRING | empty | usually wired from Prompt Batch from CSV |
| `seed` | INT | 0 | |
| `tag` | STRING | empty | |
| `params` | STRING | empty | JSON object or `key=value` lines; numbers and booleans are typed |
| `include_graph` | BOOLEAN | true | embed the API-format graph (re-queue a run from its sidecar) |
| `include_workflow` | BOOLEAN | false | advanced: embed the UI workflow as well |

Hidden inputs: `PROMPT` (the API graph) and `EXTRA_PNGINFO` (the UI workflow).

## Output

`manifest` STRING: the first image's manifest, for chaining into other nodes or a webhook.

## Sidecar schema (`jungle-fruit/run-manifest@1`)

```json
{
  "schema": "jungle-fruit/run-manifest@1",
  "created_utc": "2026-01-02T03:04:05Z",
  "tool": {"name": "comfyui-production-nodes", "version": "1.0.0"},
  "generation": {"positive": "...", "negative": "...", "seed": 1001, "tag": "cup_hero",
                 "params": {"sampler": "euler", "lut": "teal_orange", "lut_intensity": 0.75}},
  "outputs": [{"file": "batch_00001_.png", "subfolder": "jf", "batch_index": 0,
               "batch_size": 4, "width": 1024, "height": 1024}],
  "provenance": {"prompt_sha256": "…", "graph_sha256": "…", "node_count": 11,
                 "class_types": ["CLIPTextEncode", "KSampler", "..."]},
  "graph": {"…": "API-format graph, when include_graph is on"}
}
```

- `prompt_sha256` hashes the exact API graph (canonical JSON), so it identifies **the run**.
- `graph_sha256` hashes only node types and links, ignoring widget values, so all runs of
  **the same workflow** share it: group, diff or audit a delivery by workflow version.

## Privacy

Nothing about the machine is recorded: no host name, user name, absolute path or
environment variable. Only what you connect, plus the graph if `include_graph` is on.
If node values in the graph are sensitive (file names of client footage, for example),
turn `include_graph` off: the hashes are kept.
