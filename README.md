# **Learning Decision Trees as Amortized Structure Inference** #

## **Illustration** ## 
![Illustration- Learning a decision tree as a Markov decision process.](mdp.svg)

## **Initial setup** ##
For the inital setup, simply follow these steps: 

````
git clone git@github.com:MoMahfoud/RF-GFN.git
cd gfn
./setup_all.sh
````

To setup the cython files, please make sure to run:
````
cd gfn
python setup.py
````

## **Quickstart** ## 
You can train a GFlowNet to sample a single decision tree on the Iris dataset with the following command: 

````
python main.py +experiments=tree_acc
````

To make sure everything works seamlessly, you might want to use the following command for debugging: 
````
HYDRA_FULL_ERROR=1 python main.py +experiments=tree_acc
````


