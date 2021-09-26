import logging

from program.data_collect import data_collect
import warnings
import math
import os
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from glob import glob
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from typing import Any, Dict, Optional, Tuple, Union
from tokenizers import Tokenizer

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import ConcatDataset

import numpy as np

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    BertModel,
    BertPreTrainedModel,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForTokenClassification,
    DataCollatorForWholeWordMask,
    DataCollatorWithPadding,
    LineByLineTextDataset,
    LineByLineWithRefDataset,
    pipeline,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    TextDataset,
)
from transformers.modeling_outputs import MaskedLMOutput, TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertPooler




from .config import DataArguments, ModelArguments, TrainingArguments
from .fine_tune_util import DataCollatorForClassConsistency, SentenceReplacementDataset, DataCollatorForLanguageModelingConsistency, Trainer


class DeepModel(ABC):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._config: Union[PretrainedConfig, Any, None] = None
        self._model: Union[Any, None] = None
        self.tokenizer: Union[PreTrainedTokenizerBase, Any] = None

        self._model_args: ModelArguments = model_args
        self._data_args: DataArguments = data_args
        self._training_args: TrainingArguments = training_args
        self._data_collator_train = None
        self._data_collator_eval = None
        self._trainer: Trainer = None
        self._model_path = (
            self._model_args.model_name_or_path
            if self._model_args.model_name_or_path is not None and os.path.isdir(self._model_args.model_name_or_path)
            else None
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [
                -1, 0] else logging.WARN,
        )
        self._logger.setLevel(logging.INFO)

    def _load_config(self) -> None:
        if self._model_args.config_name:
            self._config = AutoConfig.from_pretrained(
                self._model_args.config_name, cache_dir=self._model_args.cache_dir)
        elif self._model_args.model_name_or_path:
            self._config = AutoConfig.from_pretrained(
                self._model_args.model_name_or_path, cache_dir=self._model_args.cache_dir)
        else:
            self._config = CONFIG_MAPPING[self._model_args.model_type]()
            self._logger.warning(
                "You are instantiating a new config instance from scratch.")
        self._config.num_labels = 10
    def _load_tokenizer(self) -> None:
        if self._model_args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._model_args.tokenizer_name, cache_dir=self._model_args.cache_dir)
        elif self._model_args.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._model_args.model_name_or_path, cache_dir=self._model_args.cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )

    @abstractmethod
    def _load_model(self) -> None:
        pass

    @abstractmethod
    def _prepare_model(self) -> None:
        pass


class MLMModel(DeepModel):
    def __init__(
            self,
            model_args: ModelArguments,
            data_args: DataArguments,
            training_args: TrainingArguments,
            vanilla_model = False
    ) -> None:
        super().__init__(model_args, data_args, training_args)
        self._language: str = ''
        self._fill_mask = None
        self._vanilla_model = vanilla_model
        self._prepare_model()

    def _load_data_collator(self) -> None:

        self.basic_loss_type =self._training_args.loss_type.split('_')[0]
        self.add_loss_type = None
        if len(self._training_args.loss_type.split('_')) > 1:
            self.add_loss_type = self._training_args.loss_type.split('_')[1]
        
        if self._config.model_type == "xlnet":
            self._data_collator = DataCollatorForPermutationLanguageModeling(
                tokenizer=self.tokenizer,
                plm_probability=self._data_args.plm_probability,
                max_span_length=self._data_args.max_span_length,
            )
        else:
            if self._data_args.mlm and self._data_args.whole_word_mask:
                self._data_collator = DataCollatorForWholeWordMask(
                    tokenizer=self.tokenizer, mlm_probability=self._data_args.mlm_probability
                )
            elif self.basic_loss_type == 'mlm' and self.add_loss_type is None:
                self._data_collator_train = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer, mlm=self._data_args.mlm, mlm_probability=self._data_args.mlm_probability
                )
                self._data_collator_eval = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer, mlm=self._data_args.mlm, mlm_probability=self._data_args.mlm_probability
                )                
            elif self.basic_loss_type == 'mlm' and self.add_loss_type is not None:
                self._data_collator_train = DataCollatorForLanguageModelingConsistency(
                    tokenizer=self.tokenizer, mlm=self._data_args.mlm, mlm_probability=self._data_args.mlm_probability
                )
                self._data_collator_eval = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer, mlm=self._data_args.mlm, mlm_probability=self._data_args.mlm_probability
                )
            elif self.basic_loss_type == 'class':
                self._data_collator_train = DataCollatorForClassConsistency(
                    tokenizer=self.tokenizer, mlm=self._data_args.mlm, mlm_probability=self._data_args.mlm_probability
                )
                self._data_collator_eval = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer, mlm=self._data_args.mlm, mlm_probability=self._data_args.mlm_probability
                )
            else:
                print("Wrong loss type")

    def _load_model(self) -> None:
        self._config.return_dict = True
        if self._model_args.model_name_or_path:
            if not self._vanilla_model:
                self._model = BertForMaskedLM.from_pretrained(
                    self._model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in self._model_args.model_name_or_path),
                    config=self._config,
                    cache_dir=self._model_args.cache_dir,
                )
            else:
                self._model = transformers.BertForMaskedLM.from_pretrained(
                    self._model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in self._model_args.model_name_or_path),
                    config=self._config,
                    cache_dir=self._model_args.cache_dir,
                )
        else:
            self._logger.info("Training new model from scratch")
            self._model = BertForMaskedLM.from_config(self._config)
        self._model.resize_token_embeddings(len(self.tokenizer))

    def _prepare_model(self) -> None:
        self._load_config()
        self._load_tokenizer()
        self._load_model()
        self._load_data_collator()

    def train(
        self,
        train_dataset: Union[LineByLineWithRefDataset,
                             LineByLineTextDataset, TextDataset, ConcatDataset],
        eval_dataset: Union[LineByLineWithRefDataset,
                           LineByLineTextDataset, TextDataset, ConcatDataset]
    ) -> None:
        self._model.train()
        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            data_collator=self._data_collator_train,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        if self._training_args.do_train:
            self._trainer.train(model_path=self._model_path)
            self._trainer.save_model()
            if self._trainer.is_world_process_zero():
                self.tokenizer.save_pretrained(self._training_args.output_dir)
        if self._training_args.do_eval:
            self._eval()

    def eval(
        self,
        eval_dataset: Union[LineByLineWithRefDataset,
                           LineByLineTextDataset, TextDataset, ConcatDataset],
        record_file: str = None,
        verbose :bool = True,
    ) -> None:
        if not verbose:
            self._training_args.disable_tqdm = True
        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            data_collator=self._data_collator_eval,
            eval_dataset=eval_dataset,
        )
        self._eval(record_file, verbose)

    def _eval(self, record_file = None, verbose=True) -> Dict:
        results = {}
        if verbose:
            self._logger.info("*** Evaluate ***")

        eval_output = self._trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        if verbose:
            self._logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                self._logger.info("  %s = %s", key, str(result[key]))

        output_eval_file = self._training_args.output_dir
        if record_file is not None:
            output_eval_file = os.path.join(output_eval_file, record_file)
        if not os.path.exists(output_eval_file):
            os.makedirs(output_eval_file)
        output_eval_file = os.path.join(output_eval_file, "eval_results_lm.txt")

        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

        return results

    def predict(self, sentence_list, batch_size=64) -> Dict:
        if self._fill_mask is None:
            self._model.eval()
            self._fill_mask = pipeline(task="fill-mask", model=self._model,
                                tokenizer=self.tokenizer, device=0, top_k=10)
        result_dict = dict()
        results = self._fill_mask(sentence_list)
        if len(sentence_list) == 1:
            results = [results]
        for i, sentence in enumerate(sentence_list):
            result_dict[sentence] = results[i]
        return result_dict


class ClassifyModel(DeepModel):
    def __init__(
            self,
            model_args: ModelArguments,
            data_args: DataArguments,
            training_args: TrainingArguments,
    ) -> None:
        super().__init__(model_args, data_args, training_args)
        self._language: str = ''
        self._fill_mask = None
        self._prepare_model()

    def _load_data_collator(self) -> None:
        self._data_collator_train = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self._data_collator_eval = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _load_model(self) -> None:
        self._config.return_dict = True
        if self._model_args.model_name_or_path:
            self._model = BertForSequenceClassification.from_pretrained(
                self._model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self._model_args.model_name_or_path),
                config=self._config,
                cache_dir=self._model_args.cache_dir,
            )
        else:
            self._logger.info("Training new model from scratch")
            self._model = BertForSequenceClassification.from_config(self._config)
        self._model.resize_token_embeddings(len(self.tokenizer))

    def _prepare_model(self) -> None:
        self._load_config()
        self._load_tokenizer()
        self._load_model()
        self._load_data_collator()

    def train(
        self,
        train_dataset: Union[LineByLineWithRefDataset,
                             LineByLineTextDataset, TextDataset, ConcatDataset],
        eval_dataset: Union[LineByLineWithRefDataset,
                           LineByLineTextDataset, TextDataset, ConcatDataset]
    ) -> None:
        self._model.train()
        self._trainer = transformers.Trainer(
            model=self._model,
            args=self._training_args,
            data_collator=self._data_collator_train,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics = compute_metrics
        )
        if self._training_args.do_train:
            self._trainer.train(model_path=self._model_path)
            self._trainer.save_model()
            if self._trainer.is_world_process_zero():
                self.tokenizer.save_pretrained(self._training_args.output_dir)
        if self._training_args.do_eval:
            self._eval()

    def eval(
        self,
        eval_dataset: Union[LineByLineWithRefDataset,
                           LineByLineTextDataset, TextDataset, ConcatDataset],
        record_file: str = None,
        verbose :bool = True,
    ) -> None:
        if not verbose:
            self._training_args.disable_tqdm = True
        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            data_collator=self._data_collator_eval,
            eval_dataset=eval_dataset,
            compute_metrics = compute_metrics
        )
        self._eval(record_file, verbose)

    def _eval(self, record_file = None, verbose=True) -> Dict:
        results = {}
        if verbose:
            self._logger.info("*** Evaluate ***")

        eval_output = self._trainer.evaluate()
        result = eval_output

        if verbose:
            self._logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                self._logger.info("  %s = %s", key, str(result[key]))

        output_eval_file = self._training_args.output_dir
        if record_file is not None:
            output_eval_file = os.path.join(output_eval_file, record_file)
        if not os.path.exists(output_eval_file):
            os.makedirs(output_eval_file)
        output_eval_file = os.path.join(output_eval_file, "eval_results_lm.txt")

        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

        return results

    def predict(self, sentence) -> Dict:
        # if self._fill_mask is None:
        #     self._model.eval()
        #     self._fill_mask = pipeline("text-classification", model=self._model,
        #                         tokenizer=self.tokenizer, device=0)
        # result_dict = dict()
        # results = self._fill_mask(sentence)
        if self._model.device.type != "cuda":
            self._model.eval()
            self._model.to("cuda:0")
        inputs = self.tokenizer(
            sentence,
            add_special_tokens=True,
            return_tensors = 'pt',
            padding=True
        )
        inputs = {
            name: tensor.to("cuda:0") if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in inputs.items()
        }
        with torch.no_grad():
            result = self._model(**inputs,output_attentions=True)
        result = {
            name: tensor.to("cpu").numpy() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in result.items()
        }
        result['tokenized_inputs'] = inputs
        return result

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class BertForMaskedLM(transformers.BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        class_labels=None,
        classification=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        class_loss = None
        if classification:
            pooled_output = self.pooler(sequence_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            if class_labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (class_labels.dtype == torch.long or class_labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        class_loss = loss_fct(logits.squeeze(), class_labels.squeeze())
                    else:
                        class_loss = loss_fct(logits, class_labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    class_loss = loss_fct(logits.view(-1, self.num_labels), class_labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    class_loss = loss_fct(logits, class_labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            output ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        else:
            output = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return output, sequence_output, class_loss
