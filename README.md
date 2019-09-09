# RecursivePlanning
research repository for recursive planning by neural net

## preparation

git clone https://github.com/advancedailab/RecursivePlanning --recursive  
cd RecursivePlanning  
./build.sh

## experiment

python3 train.py

Currently, configurations are direcly written in train.py.

## algorithms

### az.py AlphaZero

basic algorithm of combining reinforcement learning and tree search

### mctsbymcts.py

managing meta tree to conduct episode generation for steady improveness
