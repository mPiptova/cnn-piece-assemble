# Piece Assemble

This is a Python package for assembling images that have been broken into pieces (e.g., jigsaw puzzles).

This repository includes:
- Code for generating synthetic datasets of fragmented images.
- Code for training and evaluating a deep learning model that predicts matching pairs of pieces in the correct puzzle assembly. The model is based on a UNet architecture and learns which parts of puzzle piece contours should align in the correct configuration.
- Datasets used for training and evaluation (hosted on Google Drive), as well as weights of two trained models.
- Sample data and configuration files.
- Code for puzzle assembly using model predictions and a greedy algorithm with several enhancements.
- Code for evaluating the end-to-end assembly process.


## Table of Contents
- [Installation and Requirements](#installation-and-requirements)
- [Data](#data)
- [Models](#models)
- [Usage](#usage)
    - [Puzzle Generator](#puzzle-generator)
    - [Generating Training Dataset](#generating-training-dataset)
    - [Training the Model](#training-the-model)
    - [Evaluating the Model](#evaluating-the-model)
    - [Running Assembly](#running-assembly)
    - [End to End Assembly Evaluation](#end-to-end-assembly-evaluation)
- [Project Structure](#project-structure)

## Installation and Requirements 

1. Create a new virtual environment:

```bash
python3 -m venv myenv
```

2. Activate the environment:

```bash
source myenv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Install `piece_assemble` package:

```bash
pip install -e .
```

## Data

Synthetic datasets used for training and evaluation are available for download from [Google Drive](https://drive.google.com/drive/folders/1fows4Rd3V_5RoheIC9Tpe7UYMBIMYQAi?usp=sharing). The source images were obtained from the [National Gallery of Art](https://www.nga.gov/open-access-images.html).

The directory contains two nearly identical datasets, each with an explicit train/validation/test split. The only difference between them is that in one version, erosion was applied to the puzzle pieces, while in the other, it was not.

The distribution of number puzzles across the splits is shown in the table below:

| Number of Pieces | Train   | Validation | Test   | Total   |
| ---------------- | ------- | ---------- | ------ | ------- |
| 10               | 0       | 0          | 30     | 30      |
| 30               | 0       | 0          | 10     | 10      |
| 50               | 377     | 67         | 10     | 454     |
| 100              | 10      | 3          | 5      | 18      |
| 200              | 5       | 1          | 0      | 6       |
| 400              | 3       | 1          | 0      | 4       |
| **Total**        | **395** | **72**     | **55** | **522** |

## Models

In this repository, two trained models are included, placed in `models` directory:

- `UNet_20250213_180453`: Trained using non-augmented data
- `UNet_20250114_085543`: Trained using eroded data


## Usage

### Puzzle Generator

```bash
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

The following command generates a puzzle with 50 pieces with default parameters from images `path/to/image1.jpg` and `path/to/image2.jpg` and store it in `output/dir`

```bash
python generate.py 50 output/dir path/to/image1.jpg path/to/image2.jpg
```

Number of divisions, number of samples and perturbation strength values are derived from image size and number of pieces, if not set explicitly. These values provide reasonable defaults.

Following command will generate puzzle with custom parameters

```bash
python generate.py 50 \
    --num-divisions 10 \
    --num-samples 10 \
    --perturbation-strength 5 \
    output/dir path/to/image1.jpg path/to/image2.jpg
```

You can apply augmentations using the following flags:  `--color-aug` or `--erosion-strength`.

```bash
python generate.py 50 \
    --color-aug \
    --erosion-strength 5 \
    output/dir path/to/image1.jpg path/to/image2.jpg
```

The output is structured as follows:

```
output/dir/
├── <PIECE_COUNT>_image1/
│   ├── 000_mask.png
│   ├── 000.png
│   ├── ...
│   ├── original.png
│   ├── pieces.json
├── <PIECE_COUNT>_image2/
│   ├── 000_mask.png
│   ├── 000.png
│   ├── ...
│   ├── original.png
│   ├── pieces.json
```

- `<ID>.png`: Piece image, where `<ID>` is number of the piece
- `<ID>_mask.png`: Binary mask of the piece
- `original.png`: Original unfragmented image. This is not required, as it is not used for the assembly or evaluation in any way.
- `pieces.json`: The description of the correct assembly. 

`pieces.json` can look like following:
```json
{
    "transformed_pieces": [
        {
            "id": "000",
            "transformation": {
                "rotation_angle": 1.4856680401370268,
                "translation": [
                    -220.42628833974868,
                    1699.6934383814305
                ]
            }
        },
        {
            "id": "002",
            "transformation": {
                "rotation_angle": 3.155604857485427e-05,
                "translation": [
                    -0.0444450315595514,
                    0.07534405280691878
                ]
            }
        },
        {
            "id": "004",
            "transformation": {
                "rotation_angle": 4.761371918048371,
                "translation": [
                    219.98624595903345,
                    33.10453317902473
                ]
            }
        }
    ],
    "neighbors": [
        [
            "004",
            "002"
        ],
        [
            "000",
            "002"
        ]
    ]
}
```
-  `"transformed_pieces"`: All pieces that are part of the cluster, each has a piece ID and transformation.
    - `"transformation"`: A transformation that is applied to the piece in the assembly.
        - `"rotation_angle"`: Angle in radians
        - `"translation"`: Translation vector `[x, y]` in pixels. The translation should be applied after the rotation.
- `"neighbors"`: A list of pieces which are touching each other, each neighbor pair is given a list of two piece IDs.


### Generating Training Dataset

To train the model, we use datasets generated from synthetic puzzles.

This is done using the script `src/piece_assemble/dataset/create.py`:

```
python src/piece_assemble/dataset/create.py [-h] [--window-size WINDOW_SIZE] target_dir puzzle_dirs [puzzle_dirs ...]
```

The `--window-size` parameter specifies the size of the patch around each contour point used as input features. We use a `window_size` of 7 in all our experiments, which is also the default.

This script creates:
- `data_<i>.npz`: stores the input features for each piece (flattened 7x7 patches around contour points),
- `neighbors_<i>.npz`: contains ground truth similarity matrices for neighboring piece pairs,
- `data_index.csv`: maps each piece ID to the file where its input features are stored,
- `neighbors_index.csv`: maps each pair of neighboring pieces to the corresponding similarity matrix file.


Example structure of `data_index.csv`:

```
0000000,data_0.npz
0000001,data_0.npz
0000002,data_0.npz
0000003,data_1.npz
0000004,data_1.npz
...
```

- The first column lists unique piece IDs.
- The second column indicates the file containing their input features.


Example structure of `neighbors_index.csv`:

```
0000000,0000002,neighbors_0.npz
0000001,0000002,neighbors_0.npz
0000003,0000004,neighbors_2.npz
...
```

- The first two columns define the IDs of neighboring pieces.
- The third column specifies the file where their similarity matrix is stored.


Training and validation datasets should be generated separately. The recommended directory structure is:

 ```
dataset/
├── train/
│   ├── data_index.csv
│   ├── data_0.npz
│   ├── ...
│   ├── neighbors_index.csv
│   ├── neighbors_0.npz
├── val/
│   ├── data_index.csv
│   ├── data_0.npz
│   ├── ...
│   ├── neighbors_index.csv
│   ├── neighbors_0.npz
 ```

To generate the datasets, run:

```
python src/piece_assemble/dataset/create.py dataset/train dir/with/train/puzzles/*
python src/piece_assemble/dataset/create.py dataset/val dir/with/val/puzzles/*
```


The dataset can be loaded using `PairDataset` class from `piece_assemble.dataset`:

```
>> from piece_assemble.dataset import PairsDataset
>> dataset = PairsDataset(
        dataset_dir="path/to/dataset",
        seed=42,
        negative_ratio=0.1,
    )
```

`negative_ratio` specifies how many negative (non-neighboring) pairs to include, relative to the number of positive (neighboring) pairs defined in `neighbors_index.csv`.

### Training the Model

The model can be trained using the script `src/piece_assemble/tools/train.py`, which can be run as follows:

```
python src/piece_assemble/tools/train.py config.json models runs
```

where `models` is the directory where model checkpoints should be stored and `runs` is the output directory for `tensorboard`. Each model is automatically assigned an ID, such as `UNet_20250422_152254` and in `models` directory there are two versions for each model, `<MODEL_ID>_best` and `<MODEL_ID>_latest`. The configuration file is also stored for each model. 

The configuration file should have structure as follows (with recommended default values):

```json
{
    "model": {
        "embedding_dim": 128,
        "kernel_size": 3,
        "depth": 3,
        "dropout_rate": 0,
        "shared_weights": true,
        "window_size": 7,
        "background_val": -1
    },
    "train": {
        "pos_weight": 10,
        "loss_neg_ratio": null,
        "negative_ratio": 0.1,
        "negative_from_same_puzzle_ratio": 0,
        "learning_rate": 0.0001,
        "batch_size": 16,
        "epochs": 100,
        "dataset": "path/to/dataset"
    },
    "val": {
        "puzzles": [
            "path/to/puzzle1",
            "path/to/puzzle2",
            ...
        ],
        "recall_only": false,
        "threshold": 0.7
    }
}
```
- `model`
    - `embedding_dim`: Dimensionality of the output embeddings for contour points (e.g. 128-dimensional feature vectors).
    - `kernel_size`: Size of convolutional kernels used in the model (e.g. 3×3 kernels).
    - `depth`: Number of convolutional blocks (layers) used in the U-Net-like architecture.
    - `dropout_rate`: Dropout rate applied during training to prevent overfitting (0 means no dropout).
    - `shared_weights`: If `true`, only one model is trained, if `false`, two models are trained and each of them is used to process pieces in one direction (clockwise or counter-clockwise)
    - `window_size`: Size of the image patch (window) extracted around each contour point (e.g. 7x7).
    - `background_val`: Value used for pixels that doesn't belong to the piece.
- `train`
    - `pos_weight`: Weight applied to positive examples in the loss function to handle class imbalance.
    - `loss_neg_ratio`: Ratio of negative to positive examples that are used for loss computation (other values are masked). If `null`, masking is not used.
    - `negative_ratio`: Ratio of negative (non-neighboring) pairs of pieces in the training and validation dataset.
    - `negative_from_same_puzzle_ratio`: Portion of negative samples that should come from the same puzzle.
    - `learning_rate`: Learning rate used by the optimizer during training.
    - `batch_size`: Number of puzzle piece pairs processed together in one training batch.
    - `epochs`: Number of full training passes over the dataset.
    - `dataset`: Path to the directory containing the training dataset.
- `val`
    - `puzzles`: List of directories containing puzzles to be used for validation.
    - `recall_only`: If `true`, evaluation metrics only consider recall. This parameter was added to be able to evaluate the model using the [PairingNet](https://github.com/zhourixin/PairingNet/) dataset, because running the model for every possible pair of their test set would be computationally very expensive. 
    - `threshold`: Threshold to be used for the similarity matrix. 

### Evaluating the Model

The model can be evaluated using the script `src/piece_assemble/tools/eval_model.py`

```bash
python eval_model.py [-h] dataset_path puzzles_path activation_threshold models_path MODEL [MODEL ...]
```

- `dataset_path`: Path to the directory where the puzzles are stored
- `puzzles_path`: Path to the file containing the list of puzzles which should be used for evaluation. If the file contains rows `puzzle_1` and `puzzle_2`, then puzzle that will be evaluated are stored in `puzzles_path/puzzle_1` and `puzzles_path/puzzle_2`
- `activation_threshold`: Activation threshold for the model output
- `models_path`: Path to the directory where the models are stored
- `MODEL`: ID of the model to evaluate. Arbitrary number of models can be specified.

This script prints a table of results, formatted as a CSV file.

#### Example
If you want to evaluate a model on all test puzzles with 10 pieces from our dataset (see [Data](#data)), run

```bash
ls data/synth_artworks/test/ | grep ^10_ > test_10.txt
# evaluate on non-augmented dataset
python src/piece_assemble/tools/eval_model.py data/synth_artworks/test test_10.txt 0.7 models UNet_20250213_180453
# evaluate on eroded dataset
python src/piece_assemble/tools/eval_model.py data/synth_artworks_eroded/test test_10.txt 0.7 models UNet_20250213_180453
```

### Running Assembly
Images must be preprocessed beforehand.
Each piece is represented as a pair of images
`<id>.jpg` and `<id>_mask.png`, see the `data` directory for reference.

The assembly can be run using the script `src/piece_assemble/tools/run_assembly.py`.

```bash
python src/piece_assemble/tools/run_assembly.py /path/to/config
```

The output is a textual representation of clusters, where each cluster is a partial solution of the puzzle. If the assembly was successful, there is only one cluster containing all pieces. The structure is the same as `pieces.json`, for more details see [Puzzle Generator](#puzzle-generator).

This repo contains one sample configuration `sample_config.yaml`, which also contains explanation of parameters. There is also `owl_config.yaml`, which can be used to assemble
 `data/owl_101pcs` as


```bash
python src/piece_assemble/tools/run_assembly.py owl_config.yaml
```

### End to End Assembly Evaluation

Whole assembly process can be evaluated using script `src/piece_assemble/tools/eval_assembly.py`, which can be run as follows:

```bash
python src/piece_assemble/tools/eval_assembly.py [-h] config puzzles_path dataset_path
```

- `config`: configuration file with the same structure as the configuration file used for assembly. Only `img_path` param gets overridden for each puzzle that is being tested, as well as `n_iters` which is always set to 1.5 times the number of pieces.
- `puzzles_path` and `dataset_path` as in [Evaluating the Model](#evaluating-the-model)

#### Example
If you want to evaluate a model on all test puzzles with 10 pieces from our dataset (see [Data](#data)), run

```bash
ls data/synth_artworks/test/ | grep ^10_ > test_10.txt
# evaluate on non-augmented dataset
python src/piece_assemble/tools/eval_assembly.py sample_config.yaml test_10.txt data/synth_artworks/test
# evaluate on eroded dataset
python src/piece_assemble/tools/eval_assembly.py sample_config.yaml test_10.txt data/synth_artworks_eroded/test
```

### Visualizing Assembly Results

The output clusters can be visualized using the notebook `src/piece_assemble/tools/display_cluster.ipynb`. It supports displaying both the ground truth assembly and the assembly produced by our method — the only requirement is to save the output of `run_assembly.py` as a JSON file.

If the assembly is unsuccessful (i.e., incomplete), `run_assembly.py` typically outputs a list of clusters, where each cluster represents a partial solution. In that case, the output must be split into separate JSON files, as the notebook can visualize only one cluster at a time.


## Project Structure
- `src/piece_assemble/` - Main codebase (model, dataset, generator, tools).
- `data/` - Sample data and preprocessed puzzle pieces.
- `models/` - Pretrained model weights and checkpoints.