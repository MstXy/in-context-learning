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

We find the paper's practical value limited due to its training approach, which involves starting from scratch. Instead, our focus is on exploring the in-context learning capability of a text pre-trained language model, specifically GPT-2, in the context of a linear regression task. 

Given the substantial domain shift, we propose two inference approaches. The first adheres to the methodology in the referenced paper, utilizing vectorized xy-pairs as inputs. The second transforms vectorized xy-pairs into a single line of text. Our experimentation reveals that the second approach fails, while the first approach successfully learns the linear regression task. This outcome underscores the presence of valuable knowledge in text pre-trained weights that can be leveraged.

We thus further experiment with various fine-tuning techniques, including full fine-tune, soft prompting, and low rank adaptation (LoRA). Through a comprehensive comparison and analysis of results concerning computation and in-context learning errors, we demonstrate that LoRA is most effective in transferring text pre-trained knowledge to in-context linear regression learning with minimal additional computational cost.

In short, our project explores the in-context learning performance of text pre-trained GPT-2. We use five different approaches: 

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
