# Piece Assembly

This is a Python package for assembly of images broken into pieces (e.g. jigsaw puzzle).


## Usage

### Puzzle Generator

```
python src/piece_assemble/puzzle_generator/generate.py NUM_PIECES \
    [--num-divisions NUM_DIVISIONS] \
    [--num-samples NUM_SAMPLES] \
    [--perturbation-strength PERTURBATION_STRENGTH] \
    [--max-size MAX_SIZE] \
    [--erosion-strength EROSION_STRENGTH] \
    [--color-aug]
    OUTPUT_DIR \
    IMG1 [IMG2 ...]
```

Following command will generate puzzle with 50 pieces with default
parameters from images path/to/image1.jpg and path/to/image2.jpg and
store it in output/dir

```
python generate.py 50 output/dir path/to/image1.jpg path/to/image2.jpg
```

Number of divisions, number of samples and perturbation strength values are derived from
image size and number of pieces, if not set explicitly. These values provide reasonable
defaults.
Following command will generate puzzle with custom parameters

```
python generate.py 50 \
    --num-divisions 10 \
    --num-samples 10 \
    --perturbation-strength 5 \
    output/dir path/to/image1.jpg path/to/image2.jpg
```

By default, no augmentations are applied. Augmentations can be applied by setting
`--color-aug` and `--erosion-strength`.

```
python generate.py 50 \
    --color-aug \
    --erosion-strength 5 \
    output/dir path/to/image1.jpg path/to/image2.jpg
```



### Assembly
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

