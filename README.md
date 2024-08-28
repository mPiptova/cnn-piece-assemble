# Piece Assembly

This is a Python package for assembly of images broken into pieces (e.g. jigsaw puzzle).


## Usage
Images needs to be already preprocessed prior to using this project.
Each piece is represented as a pair of images
`<id>.jpg` and `<id>_mask.png`, see the `data` directory for reference.

```
python src/piece_assemble/tools/run.py /path/to/config
```

This repo contains one sample configuration `sample_config.yaml`, which also contains
explanation of parameters and can
to run the assembly of `data/owl_101pcs` as

```
python src/piece_assemble/tools/run.py sample_config.yaml
```
