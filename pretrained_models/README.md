# What's in here?

This directory contains pretrained models for predicting human
preferences over explanations for NLI and CommonsenseQA.


## URLs for pretrained models

We release six checkpoints trained on human preference feedback of
varying size and quality. These are slightly different than the models
reported in the paper, due to some implementation improvements. The
performance achieved by the 3b models is slightly better than the
average reported in the paper. The checkpoints can be downloaded here:

#### NLI

- t5-large (2.7GB): 
- t5-3b (11GB): 
- t5-11b (21GB):

Performance of these checkpoints, in comparsion to reported explanation only baseline from paper.
```
2/3
               dev    test1   test2
large-acc      57.00  59.20   59.20
3b-acc         62.00  64.00   64.00
11b-acc        64.00  69.20   66.00
expl-only-acc  51.0   51.2    50.4

large-AP       65.13  64.92   64.92
3b-AP          72.78  70.67   74.53
11b-AP         80.84  78.85   78.67
expl-only-AP   48.9   45.2    44.9

3/3
               dev    test1   test2
large-acc      35.00  34.00   34.00
3b-acc         39.00  40.80   37.20
11b-acc        39.00  44.00   38.40
expl-only-acc  30.2   30.9    27.8

large-AP       47.64  46.52   46.52
3b-AP          56.59  51.63   59.10
11b-AP         65.78  60.47   62.07
expl-only-AP   30.62  30.61   25.92
```


#### CommonsenseQA:

- t5-large (2.7GB): 
- t5-3b (11GB): 
- t5-11b (21GB):

Performance of these checkpoints, in comparsion to reported explanation only baseline from paper.
```
2/3
               dev    test1
large-acc      81.32  82.00
3b-acc         86.81  86.40
11b-acc        91.21  88.80
expl-only-acc  77.1   75.8

large-AP       81.32  85.18
3b-AP          84.60  87.23
11b-AP         86.96  88.20
expl-only-AP   75.6   77.3

3/3
               dev    test1
large-acc      50.55  56.80
3b-acc         49.45  62.80
11b-acc        54.95  63.20
expl-only-acc  42.6   47.3

large-AP       49.61  62.10
3b-AP          51.16  64.39
11b-AP         52.82  67.99
expl-only-AP   41.1   54.1
```

## How do I use the checkpoints?

If you download the above checkpoints to this directory, you can then use the two included scripts: `nli_demo.py` and `csqa_demo.py`.

For example:

```
```
