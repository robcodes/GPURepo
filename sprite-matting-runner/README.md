# sprite-matting-runner

`sprite-matting-runner` provides an end-to-end sprite-sheet matting pipeline that keeps all crops in GPU memory.  Drop pre-trained weights for FBA Matting or MatteFormer into `checkpoints/`, point the CLI at an input spritesheet and coarse binary mask, and receive stitched alpha mattes in seconds.

## Features

- Sprite-aware crop extraction with adaptive downscaling.
- Trimap generation and feature transforms mirroring the FBA Matting inference path.
- Greedy bucket batching with pixel-budget constraints and padding to CUDA-friendly shapes.
- Automatic AMP execution, optional dual-GPU splitting, and detailed timing summaries.
- Simple YAML/CLI configuration and Docker environment ready for Kaggle-style runners.

See `examples/example_config.yaml` for a quick-start configuration.


> **Note:** The public FBA Matting weights shipped by the community are licensed for non-commercial research. Provide your own checkpoint path if you require different licensing.

## FBA model setup

The runner loads `networks.models` directly from the upstream [FBA_Matting](https://github.com/MarcoForte/FBA_Matting) repository. If that package is not installed in your environment, supply the path to a local clone via either of:

- CLI flag: `--fba-repo /kaggle/working/FBA_Matting`
- Environment variable: `SMR_FBA_REPO=/kaggle/working/FBA_Matting`

The helper will temporarily prepend the provided directory to `PYTHONPATH` so the import succeeds, which avoids manual shell manipulation when running on Kaggle or other notebook hosts.

## Kaggle notebook quick start

Paste [`examples/kaggle_notebook_cell.py`](examples/kaggle_notebook_cell.py) into a
single Python cell inside your Kaggle notebook. Update the `IMG`, `MSK`, and
`SRC_WEIGHTS` paths near the top of the snippet before running it—the helper
will abort early with a clear error if the placeholder values remain or the
files are missing. The cell keeps both this repository and the upstream
`FBA_Matting` project synced to their default branches, performs an editable
install, and then launches the CLI with `SMR_FBA_REPO` set so no manual
`PYTHONPATH` tweaks are required.
