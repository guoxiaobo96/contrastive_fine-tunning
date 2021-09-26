from numpy.lib.arraysetops import isin
from sklearn.metrics.pairwise import cosine_distances
from transformers import training_args
from transformers.modeling_outputs import MaskedLMOutput
from scipy.sparse.construct import random
import numpy as np
import json
from typing import Optional, List, Dict, Tuple, Any, NewType, Union
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import os
import pickle
from packaging import version

import transformers
from transformers import BertTokenizer, BatchEncoding, PreTrainedTokenizer, DataCollatorForLanguageModeling
from transformers.file_utils import is_sagemaker_mp_enabled, is_apex_available

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class MLMConsistencyDataset(torch.utils.data.Dataset):
    def __init__(self, ori_encodings, aug_encodings):
        self.ori_encodings = ori_encodings
        self.aug_encodings = aug_encodings

    def __getitem__(self, idx):
        item = dict()
        ori_item = {key: torch.tensor(val[idx])
                for key, val in self.ori_encodings.items()}
        aug_item = {key: torch.tensor(val[idx])
                for key, val in self.aug_encodings.items()}
        item['original'] = ori_item
        item['augmentation'] = aug_item
        return item

    def __len__(self):
        return int(len(self.ori_encodings['input_ids']))


class ClassConsistencyDataset(torch.utils.data.Dataset):
    def __init__(self, ori_encodings, aug_encodings,ori_labels, aug_labels):
        self.ori_encodings = ori_encodings
        self.aug_encodings = aug_encodings
        self.ori_labels = ori_labels
        self.aug_labels = aug_labels

    def __getitem__(self, idx):
        item = dict()
        ori_item = {key: torch.tensor(val[idx])
                for key, val in self.ori_encodings.items()}
        ori_item['labels'] = torch.tensor(self.ori_labels[idx])
        aug_item = {key: torch.tensor(val[idx])
                for key, val in self.aug_encodings.items()}
        aug_item['labels'] = torch.tensor(self.aug_labels[idx])
        item['original'] = ori_item
        item['augmentation'] = aug_item
        
        return item

    def __len__(self):
        return int(len(self.ori_encodings['input_ids']))


class DataCollatorForLanguageModelingConsistency(DataCollatorForLanguageModeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # original_exampls =  deepcopy(examples)
        examples_temp = list()


        if 'original' in examples[0]:
            for example in examples:
                examples_temp.append(example['original'])
            for example in examples:
                examples_temp.append(example['augmentation'])
            examples = deepcopy(examples_temp)
        else:
            examples = examples

        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": self._collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def _collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        # Check if padding is necessary.
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result


class DataCollatorForClassConsistency(DataCollatorForLanguageModeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # original_exampls =  deepcopy(examples)
        examples_temp = list()
        class_labels = list()


        if 'original' in examples[0]:
            for example in examples:
                examples_temp.append(example['original'])
                class_labels.append(example['original']['labels'])
            for example in examples:
                examples_temp.append(example['augmentation'])
                class_labels.append(example['augmentation']['labels'])
            examples = deepcopy(examples_temp)
        else:
            examples = examples

        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": self._collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        batch["class_labels"] = torch.tensor(class_labels,dtype=torch.long)
        return batch

    def _collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        # Check if padding is necessary.
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result



class Trainer(transformers.Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._contrastive_loss = SupConLoss()
        self._cosine_loss = CosineLoss()
        self._consine_sim_loss = CosineSimLoss()

        self.basic_loss_type = self.args.loss_type.split('_')[0]
        self.add_loss_type = None
        if len(self.args.loss_type.split('_')) > 1:
            self.add_loss_type = self.args.loss_type.split('_')[1]



    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)



        if self.basic_loss_type == 'mlm' and self.add_loss_type is None:
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
        elif self.basic_loss_type == 'mlm' and self.add_loss_type is not None:
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss_consistency(model, inputs)
            else:
                loss = self.compute_loss_consistency(model, inputs)
        elif self.basic_loss_type == 'class':
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss_class_consistency(model, inputs)
            else:
                loss = self.compute_loss_class_consistency(model, inputs)            

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs, _, _ = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def compute_loss_consistency(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Except for the traditional loss, we add the cosine similarity 
        """
        self._con_loss = None

        if self.add_loss_type == 'supercon':
            self._con_loss = self._contrastive_loss
        elif self.add_loss_type == 'cosdist':
            self._con_loss = self._cosine_loss
        elif self.add_loss_type == 'cossim':
            self._con_loss = self._consine_sim_loss
            

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        size = int(inputs['input_ids'].size()[0] / 2)
        inputs_ori = dict()
        inputs_aug = dict()

        for k, v in inputs.items():
            inputs_ori[k] = inputs[k][:size]
            inputs_aug[k] = inputs[k][size:]

        outputs_ori, sequence_output_ori, _ = model(**inputs_ori)
        outputs_aug, sequence_output_aug, _ = model(**inputs_aug)
        # Save past state if it exists

        if self.args.past_index >= 0:
            self._past = outputs_ori[self.args.past_index]

        if labels is not None:
            loss_ori = self.label_smoother(outputs_ori, labels)
            loss_aug = self.label_smoother(outputs_aug, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss_ori = outputs_ori["loss"] if isinstance(outputs_ori, dict) else outputs_ori[0]
            loss_aug = outputs_aug["loss"] if isinstance(outputs_aug, dict) else outputs_aug[0]

        loss = self.args.ori_loss_scale*loss_ori + self.args.aug_loss_scale*loss_aug + self.args.con_loss_scale*self._con_loss(sequence_output_ori,sequence_output_aug)

        return (loss, outputs_ori) if return_outputs else loss

    def compute_loss_class_consistency(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Except for the traditional loss, we add the cosine similarity 
        """
        self._con_loss = None

        if self.add_loss_type == 'supercon':
            self._con_loss = self._contrastive_loss
        elif self.add_loss_type == 'cosdist':
            self._con_loss = self._cosine_loss
        elif self.add_loss_type == 'cossim':
            self._con_loss = self._consine_sim_loss
            

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        size = int(inputs['input_ids'].size()[0] / 2)
        inputs_ori = dict()
        inputs_aug = dict()

        for k, v in inputs.items():
            inputs_ori[k] = inputs[k][:size]
            inputs_aug[k] = inputs[k][size:]

        outputs_ori, sequence_output_ori, class_loss_ori = model(**inputs_ori,classification=True)
        outputs_aug, sequence_output_aug, class_loss_aug = model(**inputs_aug,classification=True)
        # Save past state if it exists

        if self.args.past_index >= 0:
            self._past = outputs_ori[self.args.past_index]

        if labels is not None:
            loss_ori = self.label_smoother(outputs_ori, labels)
            loss_aug = self.label_smoother(outputs_aug, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss_ori = outputs_ori["loss"] if isinstance(outputs_ori, dict) else outputs_ori[0]
            loss_aug = outputs_aug["loss"] if isinstance(outputs_aug, dict) else outputs_aug[0]

        sequence_output_aug = sequence_output_aug.detach()
        con_loss = self._con_loss(sequence_output_ori,sequence_output_aug)
        class_loss = class_loss_ori+class_loss_aug
        loss = self.args.ori_loss_scale*loss_ori+self.args.class_loss_scale*class_loss+self.args.con_loss_scale*con_loss

        return (loss, outputs_ori) if return_outputs else loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features_ori, features_aug, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        


        features = torch.stack([features_ori,features_aug],dim=1)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast / logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_ori, features_aug):
        loss = 1 - F.cosine_similarity(features_ori,features_aug).mean()
        return loss

class CosineSimLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_ori, features_aug):
        loss = F.cosine_similarity(features_ori,features_aug).mean()
        return loss

class SentenceReplacementDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


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

def cosine_loss(p, q):
    loss = 1 - F.cosine_similarity(p,q).mean()
    return loss