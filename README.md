# Fine-Tuning Transformers to In-Context Learn Simple Function Classes.

### CS 182/282A Deep Learning Final Project

#### Project Team:
- William Zhang, EECS Graduate, [chengyu_zhang@berkeley.edu](mailto:chengyu_zhang@berkeley.edu)
- Yuanbo Chen, EECS Graduate, [yuanbo_chen@berkeley.edu](mailto:yuanbo_chen@berkeley.edu)
- Eric Tai, EECS Graduate, [erictai@berkeley.edu](mailto:erictai@berkeley.edu)
- Michelle Wang, EECS Undergraduate,  [michellew@berkeley.edu](mailto:michellew@berkeley.edu)

<br>

Our project extends from the paper: <br>
**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br>

```bibtex
    @InProceedings{garg2022what,
        title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
        author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
        year={2022},
        booktitle={arXiv preprint}
    }
```

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
    python eval.py models
    ```
    Or use `eval.ipynb`.

5. To train, in `conf/wandb.yaml`, provide wandb user name for `entity`. Then for different task, run:

    ```
    ## Task: linear regression
    python train.py --config conf/linear_regression.yaml

    ## Task: sparse linear regression
    python train.py --config conf/sparse_linear_regression.yaml
    ```
