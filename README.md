# Fine-Tuning Text Pre-Trained Transformers to In-Context Learn Simple Function Classes.

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

Our project explores the in-context learning performance of text pre-trained GPT-2. We use five different approaches: 

- infer from text pre-trained weights with default (vectorized) input,
- infer from text pre-trained weights with devised text input,
- full fine-tune,
- fine-tune with soft prompting,
- fine-tune with low rank adaptation (LoRA). 

We focus primarily on the simple linear regression task to compare the results. 

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

3. Download [model checkpoints](https://drive.google.com/file/d/1zuWgKbmWAj8GpyLmU_cg04iV4u4TUfNi/view?usp=sharing) and extract them in the current directory under `models` folder.

    ```
    unzip models.zip
    ```

4. To evaluate, use `eval.ipynb`, each step is already in the notebook. 
> **Note**: all metrics have already been computed by the end of each training, so we removed `state.pt` in the folder to save space. 

5. To train, in `conf/wandb.yaml`, provide wandb user name for `entity`. Then for different task, run:

    ```
    cd src

    ## Approach: only pre-train
    python train.py --config conf/finetune_baseline_linear_regression.yaml
    
    ## Approach: full fine-tune
    python train.py --config conf/finetune_unfreeze_linear_regression.yaml

    ## Approach: soft prompting fine-tune
    python train.py --config conf/finetune_softprompt_linear_regression.yaml

    ## Approach: LoRA fine-tune
    python train.py --config conf/finetune_lora_linear_regression.yaml

    ## Approach: train from scratch
    python train.py --config conf/linear_regression.yaml

    ## Approach: text input from pre-trained weights
    python train.py --config conf/finetune_textbase_linear_regression.yaml
    # note this approach fails to generate numerical output, so will throw out error.
    # but it will be fun to run it in debug mode to check the outputs.
    # vscode launch.json file is attached. 
    ```
