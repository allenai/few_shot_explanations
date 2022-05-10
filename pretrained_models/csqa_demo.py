'''
A (hopefully) Simple API for serving explanation score requests.

input_string = (
    f"{question} answer: {gold_label}. "
    + f" explanation: {abstr_expl}."
)

here are some example input strings:

If you feel like everything is spinning while climbing you are experiencing what? answer: vertigo. explanation: Vertigo is often experienced while climbing or at heights.
Where do you get clothes in a shopping bag? answer: retail store. explanation: For any large item where convenience is beneficial, one might go to a retail store, either a regular one or a big-box store like walmart.
Where should a cat be in a house? answer: floor. explanation: A cat should be on the floor, not on a rug.
'''

import argparse
import torch
import transformers
import os
import tqdm
import numpy as np

_model, _tokenizer = None, None

model2url = {
    'large': 'https://storage.googleapis.com/ai2-mosaic-public/projects/few-shot-explanations/pretrained_models/commonsense_qa/valloss%3D0.28665~model%3Dt5-large~lr%3D0.0001~seed%3D1~labelagg%3D0_just_weights.pt',
    '3b': 'https://storage.googleapis.com/ai2-mosaic-public/projects/few-shot-explanations/pretrained_models/commonsense_qa/valloss%3D0.28925~model%3Dt5-3b~lr%3D0.0001~seed%3D1~labelagg%3D0_just_weights.pt',
    '11b': 'https://storage.googleapis.com/ai2-mosaic-public/projects/few-shot-explanations/pretrained_models/commonsense_qa/cose_deepspeed_valloss%3D0.00000~model%3Dt5-11b~lr%3D0.00001~seed%3D1~labelagg%3D0.pt',
}

def get_model(model_type, device=None):
    global _model, model2url
    if model_type not in {'11b', '3b', 'large'}:
        raise NotImplementedError('{} is not a valid model please use "3b" or "large"'.format(model_type))

    if _model is None:
        hf_model_name = 't5-' + model_type
        print('Loading model: this will run only once.')

        if model_type == 'large':
            model_path = 'valloss=0.28665~model=t5-large~lr=0.0001~seed=1~labelagg=0_just_weights.pt'
        elif model_type == '3b':
            model_path = 'valloss=0.28925~model=t5-3b~lr=0.0001~seed=1~labelagg=0_just_weights.pt'
        elif model_type == '11b':
            model_path = 'cose_deepspeed_valloss=0.00000~model=t5-11b~lr=0.00001~seed=1~labelagg=0.pt'

        if not os.path.exists(model_path):
            print('Please download weights for {} model and put in current directory.'.format(model_path))
            print('for example, wget {}'.format(model2url[model_type]))
            quit()

        state = torch.load(model_path)
        if 'model_state_dict' in state:
            state = state['model_state_dict']

        _model = transformers.AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
        if model_type == '11b': # need to resize due to deepspeed, these entires are not accessed.
            _model.resize_token_embeddings(len(transformers.AutoTokenizer.from_pretrained(hf_model_name)))
        _model.load_state_dict(state)
        _model.eval()
        if device is not None:
            _model = _model.to(device)

    return _model


def get_tokenizer(model_type):
    global _tokenizer
    if model_type not in {'3b', 'large', '11b'}:
        raise NotImplementedError('{} is not a valid model please use "3b" or "large" or "11b"'.format(model_type))

    if _tokenizer is None:
        hf_model_name = 't5-' + model_type
        _tokenizer = transformers.T5TokenizerFast.from_pretrained(hf_model_name)

    return _tokenizer


class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        res = self.tokenizer(self.data[idx]['input'], truncation=True)
        res['labels'] = self.tokenizer(self.data[idx]['label']).input_ids
        return res

    def __len__(self):
        return len(self.data)


def get_scores(inputs, model_type, device=None, batch_size=32, verbose=False):
    '''
    Inputs:
      - a list of explanations to score, e.g.,:
        premise: A man getting a tattoo on his back. hypothesis: A woman is getting a tattoo. answer: contradiction. explanation: Because the tattoo artist is a man, the person getting the tattoo is not a woman.
      - model type, either "3b" or "large" or "11b"
      - device: which torch device to load model on, e.g., "cuda:3"
    Outputs:
      - P(good explanation); higher is better
    '''
    assert model_type in {'large', '3b', '11b'}

    if isinstance(inputs, str):
        inputs = [inputs]

    model = get_model(model_type, device=device)
    tokenizer = get_tokenizer(model_type)

    score_itr = T5Dataset([{'input': inp, 'label': 'x'} for inp in inputs], tokenizer) # dummy labels for inference
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        return_tensors='pt'
    )
    score_itr = torch.utils.data.DataLoader(score_itr, shuffle=False, collate_fn=data_collator, batch_size=batch_size)
    score_itr = score_itr if not verbose else tqdm.tqdm(score_itr, total=len(score_itr))

    good_idx, bad_idx = tokenizer('good').input_ids[0], tokenizer('bad').input_ids[0]
    scores = []
    with torch.no_grad():
        for batch in score_itr:
            if device is not None:
                input_ids, attention_mask, targets = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            model_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            logits_pos = model_output['logits'][:, 0, good_idx].cpu().numpy()
            logits_neg = model_output['logits'][:, 0, bad_idx].cpu().numpy()
            exp_logit_pos, exp_logit_neg = np.exp(logits_pos), np.exp(logits_neg)
            scores.extend(list([float(x) for x in exp_logit_pos / (exp_logit_pos + exp_logit_neg)]))
    return scores


def parse_args():
    '''
    Optional args for main function, mostly just to test.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_type',
        default='large',
        choices={'large', '3b', '11b'})
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)

    scores = get_scores(
        ['If you feel like everything is spinning while climbing you are experiencing what? answer: vertigo. explanation: Vertigo is often experienced while climbing or at heights.',
         'Where do you get clothes in a shopping bag? answer: retail store. explanation: For any large item where convenience is beneficial, one might go to a retail store, either a regular one or a big-box store like walmart.',
         'Where should a cat be in a house? answer: floor. explanation: A cat should be on the floor, not on a rug.'],
        args.model_type,
        device=args.device,
        batch_size=args.batch_size,
        verbose=False)
    print(scores)

    # t5-large:
    # [0.8819078803062439, 0.1019817441701889, 0.44061794877052307]

    # t5-3b:
    # [0.7853091955184937, 0.11382164061069489, 0.5468584299087524]

    # t5-11b:
    # [0.8240450024604797, 0.047063715755939484, 0.20420850813388824]


if __name__ == '__main__':
    main()
