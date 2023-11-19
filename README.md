# Fine-Tuning Transformers to In-Context Learn Simple Function Classes.

### CS 182/282A Deep Learning Final Project

#### Project Team:
- William Zhang, EECS Graduate, [chengyu_zhang@berkeley.edu](mailto:chengyu_zhang@berkeley.edu)
- Yuanbo Chen, EECS Graduate, [yuanbo_chen@berkeley.edu](mailto:yuanbo_chen@berkeley.edu)
- Eric Tai, EECS Graduate, [erictai@berkeley.edu](mailto:erictai@berkeley.edu)
- Michelle Wang, EECS Undergraduate,  [michellew@berkeley.edu](mailto:michellew@berkeley.edu)


Our project extends from the paper: <br>
**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br>

Our project explores the in-context learning performance of text pre-trained GPT-2. We then fine-tune using techniques such as Soft Prompting, focusing primarily on the simple linear regression task to compare the results. We also extends to m-degree polynomial functions, as well as sparse linear regression and decision trees. 

## Getting started
1. Clone the repository
    ```
    git clone https://github.com/MstXy/in-context-learning.git
    ```

2. Install the dependencies using Conda.

    ```
    conda env create -f environment.yml
    conda activate icl
    ```

3. Download [model checkpoints](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip) and extract them in the current directory.

    ```
    wget https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip
    unzip models.zip
    ```

4. To evaluate:
    ```
    cd src
    python eval.py ../models
    ```
    Or use `eval.ipynb`.

5. To train, in `conf/wandb.yaml`, provide wandb user name for `entity`. Then for different task, run:

    ```
    cd src

    ## Task: linear regression
    python train.py --config conf/linear_regression.yaml

    ## Task: sparse linear regression
    python train.py --config conf/sparse_linear_regression.yaml
    ```
