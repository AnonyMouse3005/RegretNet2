# RegretNet
This repository reproduces the paper [Deep Learning for Multi-Facility Location Mechanism Design](https://econcs.seas.harvard.edu/files/econcs/files/golowich_ijcai18.pdf).

## TODO
- [x] Implement the rules: percentile, dictator, constant and non-SP
- [x] MoulinNet malfunctions when k>1. 
- [x] MoulinNet does not converge to the social cost as in the paper for k=3
- [x] RegretNet malfunctions. It needs an **augmented Lagrangian solver** to optimize, which will cost some time to implement.
- [x] Implement max regret estimation in `RegretNetSystem`

## Installation
1. Install python > 3.6
2. Setup a virtual environment
```shell
virtualenv venv
# Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```
3. Install pytorch according to the [official website](https://pytorch.org/get-started/locally/)
```shell
# Example: Install pytorch with cuda v11.6 on Windows via pip 
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
4. Install pytorch-lightning
```shell
pip install pytorch-lightning
```
5. Install jsonargparse
```shell
pip install jsonargparse[signatures]
```
6. Install defaultlist
```shell
pip install defaultlist
```

## Usage
### Grid Search (train and test model with different n, k, d)
- use [run_search.py](run_search.py). Change parameters in the beginning of `main()`.
```shell
python run_search.py
```
### Interactive (train one model and test with user input)
- use [run_interactive.py](run_interactive.py). Change parameters in the beginning of `main()`.
```shell
python run_interactive.py
```
### View train logs (only by models trained in [run_interactive.py](run_interactive.py))
- use [run_tensorboard.py](run_tensorboard.py). Open the link in the browser to view plots.
```shell
python run_tensorboard.py
```
