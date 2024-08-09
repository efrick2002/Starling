# Nectar

This codebase contains the scripts used to collect the prompts in [berkeley-nest/Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar) as well as generate the seven-wise comparisons.  In addition, we include data from ablations and the corresponding Jupyter notebooks to visualize the experimental data.

## Setup

The code has been verified on Python 3.10, but other versions of Python are likely compatible.

Simply run the command below to download required packages:

```bash
pip install -r requirements.txt
```

## Collecting Prompts

Code for collecting prompts is found in the [prompts](./prompts/) directory.  Public datasets were compiled into one dataset via [create_data.ipynb](prompts/create_data.ipynb), including some basic heuristics for duplicate detection.

The [prompts/visualizations](./prompts/visualizations/) directory contains the Jupyter notebook used to generate figures for the paper.

## Distillation

Code for distillation is found in the [distillation](./distillation/) directory.  [Distill.py](./distillation/distill.py) contains the script used to inference various models in parallel.

## Ratings

Code for inferencing the seven-wise ratings is found in the [rating](./ratings/) directory.  Specially, [rate.py](./ratings/rate.py) contains the final code used to generate the Nectar ratings for all 180k rows.

The [ratings/visualizations](./ratings/visualizations/) directory contains variance Jupyter Notebooks to generate visualizations.  The data for these visualizations can be found in the [rating/results](./ratings/results/) directory.  Inside each directory in the results directory has a `prompt_log.txt` containing the prompt used, `log_args.txt` containing the rating script args used, and also `rankings.jsonl` containing the outputted rankings from the experiment.

The [ratings/experiments](./ratings/experiments/) folder contains some extra scripts specifically for running certain experiments.  Their outputs are found in the directories contained in [rating/results](./ratings/results/) with the associated names.

- **[measure_k_position_bias.py](./ratings/experiments/measure_k_positional_bias.py)**: Measures how positional bias changes as K increases.
  
- **[measure_k_to_pairwise.py](./ratings/experiments/measure_k_to_pairwise.ipynb)**: Measures how judgment agreement with pairwise ratings changes as K increases.
  
- **[rate_pointwise.py](./ratings/experiments/rate_pointwise.py)**: Creates pointwise ratings instead of pairwise ratings.

- **[rate_verbose.py](./ratings/experiments/rate_verbose.py)**: Tests ratings with more explicit anti-verbosity prompting.
  