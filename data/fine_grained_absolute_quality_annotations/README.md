## Overview

The data in this directory includes the fine-grained annotations used to produce Figures 2 and 3.

For each task (SNLI or CommonsenseQA), the same 250 test instances are annotated regardless of explanation source. There are five files, each representing a different explanation source:
- For CommonsenseQA, there are explanations from CoS-E, ECQA, and GPT-3 with greedy sampling.
- For SNLI, there are explanations from e-SNLI and GPT-3 with greedy sampling.  

Each line in the csv files indicates the judgements of one annotator (thus, each dataset instance has 3 lines, representing 3 distinct annotators). We have removed worker identifiers to preserve anonymity and to prevent reconstruction of individual annotator judgements. Please note that acceptability annotations were collected later and merged into the file, meaning that they are most likely provided by different annotators than the other 5 annotations.

## File Coding

Aligning the numerical values in the csv files to the y-axis in Figures 2/3 and the user interface shown in Appendix Figure 8, the values represent the following:
- Generality score: computed from `Answer.generality_factuality`. A value of -9 means the user selected the choice "needs more information to judge factuality". We thus treat the explanation as "specific" and assign -1 as the generality score. A value of -1, 0, or 1 means the user was able to judge the explanation's factuality. We thus treat the explanation as "general" and assign 1 as the generality score.
- Factuality score: also computed from `Answer.generality_factuality`. If a user selected -9, the factuality score is undefined as it cannot be assessed without more information. Value of -1 corresponds to "generally false", 0 to "sometimes or partially true", and 1 to "generally true".
- `Answer.grammar`: 1 indicates the user judged the statement to be grammatical and -1 the opposite.
- `Answer.new_info`. 1 indicates the user judged the explanation to contains new facts, information, or reasoning not stated in the task input; -1 the opposite.
- `Answer.supports_label`: 1 indicates the user judged the new info provided to be relevant to justifying the gold label; -1 the opposite.
- `Answer.amount_info`: -1 indicates the user judged the explanation to have "not enough info", 0 "enough info", and 1 "too much info".
- `Answer.acceptable`: 1 indicates acceptable; -1 not.

Note that some fields are empty; this is because some questions (such as Amount Info) were only asked to the annotator if they selected "yes" to a previous question (e.g., that the explanation contains new info). See Appendix B.3 for more details.

## Getting acceptability annotations

To obtain the acceptability values in the Figures for GPT-3-generated explanations, these files can be merged with those in the `acceptability_data` directory on the `Input.id` field. For example, to obtain GPT-3 greedy explanations' acceptability scores for the CommonsenseQA test set, the `commonsenseqa/gpt3_greedy.csv` file given here can be merged with `acceptability_annotations/commonsenseqa_test.csv` on `Input.id`, locating the `Input.source` id of the greedy explanation (1-5) in that file, and then determining if that number is in the pipe-delimited list of annotator-selected acceptable explanations. Please note that the acceptability annotations are not necessarily provided by the same annotators as those who provided the other fine-grained annotations.
