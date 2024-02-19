# Researching Switch Transformers
### Looking to solve open problems in Switch Tranformer fine-tuning and inference.

### Running Code

#### Import Conda Environment

> `conda env create -f env.yml`

#### Run python code

> `conda activate switch`

> `python switch_transformers.py`


### Research

#### Possible topics:
- Distilling large switch transformers into a small dense *or* sparse network for faster finetuning and/or inference
- Enabling extreme quantinization on the large switch transformers by finetuning with a clipped softmax function in the attention heads https://arxiv.org/abs/2306.12929
- Studying switch transformer performance in Computer Vision tasks

#### Papers to Read and Summarize for Related Works:
(from https://huggingface.co/blog/moe)
1. [Adaptive Mixture of Local Experts (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
2. [Learning Factored Representations in a Deep Mixture of Experts (2013)](https://arxiv.org/abs/1312.4314)
3. [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017)](https://arxiv.org/abs/1701.06538)
4. [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Jun 2020)](https://arxiv.org/abs/2006.16668)
5. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Jan 2022)](https://arxiv.org/abs/2101.03961)
6. [ST-MoE: Designing Stable and Transferable Sparse Expert Models (Feb 2022)](https://arxiv.org/abs/2202.08906)
7. [Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models (May 2023)](https://arxiv.org/abs/2305.14705)