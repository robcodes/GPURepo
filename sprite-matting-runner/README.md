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
