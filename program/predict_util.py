from scipy.sparse.construct import random
import numpy as np
from typing import List
import torch
from torch.utils.data import  Dataset
import os
import pickle

from transformers import BatchEncoding,  PreTrainedTokenizer



class MaksedPredictionDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, sentence_list: List[str], cached_features_file='', overwrite_cache=False):

        if os.path.exists(cached_features_file) and not overwrite_cache:
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)

        self.examples = tokenizer.batch_encode_plus(
            sentence_list, max_length=512, padding=True)["input_ids"]

        # self.examples = []
        # for sentence in sentence_list:
        #     tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
        #     self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text))

        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def encode_scores(scores, encodings: BatchEncoding):
    scores = [[score for score in doc] for doc in scores]
    encoded_scores = []
    input_ids = list()
    token_type_ids = list()
    attention_mask = list()
    encoding_list = list()
    error_count = 0

    for (doc_scores, doc_input_ids, doc_attemtion_mask, doc_token_type_ids, doc_offset, doc_encoding) in zip(scores, encodings.input_ids, encodings.attention_mask, encodings.token_type_ids, encodings.offset_mapping, encodings._encodings):
        # create an empty array of -100
        try:
            doc_enc_scores = np.ones(len(doc_offset), dtype=float) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_scores[(arr_offset[:, 0] == 0) & (
                arr_offset[:, 1] != 0)] = doc_scores
            encoded_scores.append(doc_enc_scores.tolist())
            input_ids.append(doc_input_ids)
            token_type_ids.append(doc_token_type_ids)
            attention_mask.append(doc_attemtion_mask)
            encoding_list.append(doc_encoding)
        except:
            error_count += 1

    data = {'input_ids': input_ids, 'token_type_ids': token_type_ids,
            'attention_mask': attention_mask}
    encodings = BatchEncoding(data, encoding_list)

    return encoded_scores, encodings
