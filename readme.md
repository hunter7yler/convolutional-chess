# CNN Chess Engine

This project tests the efficacy of Convolutional Neural Networks (CNNs) in Chess AIs.

### Setup

This project uses Python 3. All other dependencies are listed in requirements.txt. To get a virtualenv set up for this project, run the following three commands from the nn_chess directory:

Create a virtualenv:

```
python3 -m venv ./nnc_env
```
Activate the virtualenv:
```
source ./nnc_env
```
Install all dependencies:
```
pip install -r requirements.txt
```

### CNN Training

Replace the sample file nn_chess/data/data.pgn with any data you like. I recommend a year's worth of data from the [FICS Games Database](https://www.ficsgames.org/download.html). 

Now you can train your CNN by executing the following: 
```
python nn_gen.py
```
This will save two CNNs into nn_chess/data, one for recommending what piece to move and another for deciding where to put it.

### Run Simulations

Once you've trained your CNNs, you can test the AIs against the other basic chess engines using Jupyter Notebook. The cells in local_engines.py are examples tests comparing the CNN AIs against the basic chess engines from the Chessmate Python module.