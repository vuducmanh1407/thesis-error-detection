# Thesis: Improving Error Detection in Procedural Tasks using Interaction Awareness

This repository contains the implementation of my Master's Thesis. The code is based on the implementation of [GTG2Vid](https://github.com/robert80203/GTG2Vid).

## Installation

Please install the dependencies provided in the ```requirements.txt``` file.


## Training and evaluation

To train the model, run:
```bash
python main.py --config {PATH/TO/CONFIGS.json} --dir {OUTPUT_FOLDER} --seed {SEED}  
```
The provided configuration file can be found in the ```configs``` folder.

To evaluate, run
```bash
python main.py --config {PATH/TO/CONFIGS.json} --dir {FOLDER_TO_EVALUATE} --eval --vis
```.
