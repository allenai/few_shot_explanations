Each line in the csv files indicates the judgements of one annotator (thus, each dataset instance has 3 lines, representing 3 distinct annotators). 
We have removed worker identifiers to preserve anonymity and to prevent reconstruction of individual annotator judgements.

The `Answer.acceptable` field presents which of the 5 explanations presented (listed in column `Input.explanation_{1-5}`) are judged to be acceptable, pipe-separated.
For example, `1|3|5` would indicate the user selected explanations 1, 3, and 5 as acceptable. 
An empty string means none of the explanations were selected.

The explanations are in a shuffled order. `Input.source_sample_{1-5}` indicates which of {greedy, samples 1-4} the explanations are, and `Input.nll_{1-5}` their normalized negative-log likelihoods (as measured by GPT-3), which are used to compute the NLL baseline in the paper.