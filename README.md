# Starling-7B: Improving Helpfulness and Harmlessness with RLAIF

This is the code base for the Starling project from UC Berkeley including:

- Paper: Soon
- Blog Post: [starling.cs.berkeley.edu](https://starling.cs.berkeley.edu/)
- LLM: [Starling-LM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha)
- RM: [Starling-RM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha)
- RM: [Starling-RM-34B](https://huggingface.co/Nexusflow/Starling-RM-34B)
- Dataset: [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar)

We include code for the full pipeline: from dataset curation to reward model training to PPO finetuning.

The code base is split into 3 parts:

1. [Nectar](./nectar): All code pertaining to dataset curation, including prompt sourcing, response distillation, and judgment curation.
2. [Reward Model Training](./reward-model-training/): All code pertaining to reward model training using the [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar) dataset.
3. [trlx](./trlx/): All code pertaining to PPO finetuning, a customized fork of the original [trlx](https://github.com/CarperAI/trlx) codebase.*

Each part has its own respective documentation.

<sub><sup>\*Note that it seems the trlx codebase is no longer maintained. Parts of the code may be outdated or may not be compatible with newer systems.<sub><sup>

## Citation

```
@misc{
starling2024,
title = {Starling-7B: Improving Helpfulness and Harmlessness with RLAIF},
author = {Zhu, Banghua and Frick, Evan and Wu, Tianhao and Zhu, Hanlin and Ganesan, Karthik and Chiang, Wei-lin and Zhang, Jian and Jiao, Jiantao},
booktitle = {First Conference on Language Modeling},
year = {2024},
}
```

