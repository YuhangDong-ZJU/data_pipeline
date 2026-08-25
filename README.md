# DROID data pipeline

Small, restartable utilities for preparing DROID-derived annotations. The
repository is organized by annotation modality so each workflow can be copied,
tested, and deployed independently.

## Repository layout

```text
data_pipeline/
├── droid_depth/    # FoundationStereo depth post-processing utilities
└── droid_normals/  # NormalCrafter video-normal annotation pipeline
```

## Workflows

### DROID depth

[`droid_depth/`](droid_depth/) contains tools for moving completed depth
episodes into a LeRobot dataset, packing depth PNG directories into TAR shards,
and removing padded tail timesteps identified by FoundationStereo metadata.

Start with [`droid_depth/README.md`](droid_depth/README.md).

### DROID normals

[`droid_normals/`](droid_normals/) contains the NormalCrafter worker, launcher,
and the pinned upstream patch used to generate temporally consistent normal-map
MP4 annotations.

Start with [`droid_normals/README.md`](droid_normals/README.md).

## Repository scope

Dataset contents, generated annotations, model checkpoints, third-party source
trees, and Python environments are intentionally kept outside this repository.
Each workflow documents its own external dependencies and server defaults.
