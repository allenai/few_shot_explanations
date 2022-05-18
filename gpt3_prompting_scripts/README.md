## Requirements
- Huggingface `datasets` (for CoS-E v1.11 and E-SNLI datasets)
- `openai`
- `gitpython`
- `jsonlines`
- You will need to set the `OPENAI_API_KEY` environment variable using your OpenAI credentials.
- If using ECQA explanations, you will need to download `ecqa.jsonl` from [https://github.com/dair-iitd/ECQA-Dataset](https://github.com/dair-iitd/ECQA-Dataset) and place it in this directory.
- All code is formatted using [`black`](https://github.com/psf/black).

## To produce predictions
`python get_[snli,cose]_gpt3_completion_accuracy.py --task_type [explanationOnly, labelOnly, jointBoth] --model [davinci, curie, ada, babbage] --num_samples [1, 5, etc.] --prime_source [ours, dataset] --split [train, validation, test] --dataset [snli, cose_v1.11, ecqa] --total_train_instances [integer] --get_shuffled_examples`
- `task_type`: `explanationOnly` (predict explanation), `labelOnly` (predict label), `jointBoth` (predict both label and explanation)
- `num_samples`: `1` for greedy, else more if also want stochastic samples (`5` used in paper). The greedy sample is always included, so # stochastic samples will be num_samples-1.
- `total_train_instances`: optional; only needs to be specified if `split`=`train`. Value is 1000 in the paper.
- `get_shuffled_examples`: including this flag will also create a file of GPT-3 vs. dataset explanations in shuffled order, which can be used as input for a head-to-head crowdsourcing study.
- Other optional arguments (`stop_token`, `generation_length`, `temperature`, `top_p`, `random_seed`, `frequency_penalty`, `presence_penalty`, etc.) control the GPT-3 generation process and are set to the default values in the scripts.

### Examples:
- Generate SNLI greedy explanations conditioned on **dataset** explanations and gold labels: `python get_snli_gpt3_completion_accuracy.py --task_type explanationOnly --model davinci --num_samples 1 --prime_source ours --split test`
- Generate SNLI greedy explanations conditioned on **author-written** explanations and gold labels: `python get_snli_gpt3_completion_accuracy.py --task_type explanationOnly --model davinci --num_samples 1 --prime_source dataset --split test`
- Generate SNLI label predictions for prediction-accuracy-wise analysis (Appendix C): `python get_snli_gpt3_completion_accuracy.py --task_type labelOnly --model davinci --num_samples 1 --prime_source dataset --split test`
- Generate 1 greedy + 4 stochastic samples for filter model training data: `python get_snli_gpt3_completion_accuracy.py --task_type explanationOnly --model davinci --num_samples 5 --prime_source ours --split train --total_train_instances 1000`
