# Thesis: Improving Error Detection in Procedural Tasks using Interaction Awareness

This repository contains the implementation of the Master's Thesis. The code is based on the implementation of [GTG2Vid](https://github.com/robert80203/GTG2Vid).

## Installation

Please install the dependencies provided in the ```requirements.txt``` file. 

## Dataset

For the datasets used, please refer to [GTG2Vid](https://github.com/robert80203/GTG2Vid).

To obtain the pre-extracted object features, please copy the file ```extract_dinov3_features_{DATASET}.py``` to the main folder of the dataset used. Then you can extract features as:

```bash
python extract_dinov3_features_{DATASET}.py -d {TASK_NAME}  
```

where ```TASK_NAME``` is the activity to be extracted.

## Training and evaluation

To train the model, run:
```bash
python main.py --config {PATH/TO/CONFIGS.json} --dir {OUTPUT_FOLDER} --seed {SEED}  
```
The provided configuration file can be found in the ```configs``` folder.

To evaluate, run
```bash
python main.py --config {PATH/TO/CONFIGS.json} --dir {FOLDER_TO_EVALUATE} --eval --vis
```
