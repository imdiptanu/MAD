from farm.data_handler.processor import Processor
from tokenizers.pre_tokenizers import WhitespaceSplit
from farm.data_handler.samples import (
    Sample,
    SampleBasket,
)
from farm.data_handler.utils import expand_labels
import ast
import numpy as np
import pandas as pd


class MTLProcessor(Processor):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_filename,
        test_filename,
        delimiter,
        dev_split=0.15,
        dev_filename=None,
        label_list=None,
        metric=None,
        proxies=None,
        **kwargs,
    ):
        self.delimiter = delimiter

        super(MTLProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )

    def file_to_dicts(self, file: str):
        dicts = list()
        df = pd.read_csv(file)
        for text, label, tokens in zip(
            df.post_tokens.values, df.post_label.values, df.toxic_tokens.values
        ):
            columns = dict()
            text = ast.literal_eval(text)
            tokens = ast.literal_eval(tokens)
            columns["text"] = " ".join(text)
            columns["document_level_task_label"] = label  # Key hard-coded
            columns["token_level_task_label"] = list(map(str, tokens))  # Key hard-coded
            dicts.append(columns)
        return dicts

    @staticmethod
    def _get_start_of_word(word_ids):
        words = np.array(word_ids)
        words[words == None] = -1
        start_of_word_single = [0] + list(np.ediff1d(words) > 0)
        start_of_word_single = [int(x) for x in start_of_word_single]
        return start_of_word_single

    # Most of the code is copied from NERProcessor - dataset_from_dicts()
    def dataset_from_dicts(
        self, dicts, indices=None, return_baskets=False, non_initial_token="X"
    ):
        self.baskets = []
        self.pre_tokenizer = WhitespaceSplit()

        texts = [x["text"] for x in dicts]
        words_and_spans = [self.pre_tokenizer.pre_tokenize_str(x) for x in texts]
        words = [[x[0] for x in y] for y in words_and_spans]

        word_spans_batch = [[x[1] for x in y] for y in words_and_spans]

        tokenized_batch = self.tokenizer.batch_encode_plus(
            words,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            is_split_into_words=True,
        )

        for i in range(len(dicts)):
            tokenized = tokenized_batch[i]
            d = dicts[i]
            id_external = self._id_from_dict(d)
            if indices:
                id_internal = indices[i]
            else:
                id_internal = i

            input_ids = tokenized.ids
            segment_ids = tokenized.type_ids
            initial_mask = self._get_start_of_word(tokenized.words)
            assert len(initial_mask) == len(input_ids)

            padding_mask = tokenized.attention_mask

            if return_baskets:
                token_to_word_map = tokenized.words
                word_spans = word_spans_batch[i]
                tokenized_dict = {
                    "tokens": tokenized.tokens,
                    "word_spans": word_spans,
                    "token_to_word_map": token_to_word_map,
                    "start_of_word": initial_mask,
                }
            else:
                tokenized_dict = {}

            feature_dict = {
                "input_ids": input_ids,
                "padding_mask": padding_mask,
                "segment_ids": segment_ids,
                "initial_mask": initial_mask,
            }

            for task_name, task in self.tasks.items():
                try:
                    label_name = task["label_name"]
                    labels_word = d[label_name]
                    label_list = task["label_list"]
                    label_tensor_name = task["label_tensor_name"]

                    if task["task_type"] == "classification":
                        label_ids = [label_list.index(labels_word)]
                    elif task["task_type"] == "ner":
                        labels_token = expand_labels(
                            labels_word, initial_mask, non_initial_token
                        )
                        label_ids = [label_list.index(lt) for lt in labels_token]
                except ValueError:
                    label_ids = None
                    problematic_labels = set(labels_token).difference(set(label_list))
                    print(
                        f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                        f"\nWe found a problem with labels {str(problematic_labels)}"
                    )
                except KeyError:
                    label_ids = None
                if label_ids:
                    feature_dict[label_tensor_name] = label_ids

            curr_sample = Sample(
                id=None, clear_text=d, tokenized=tokenized_dict, features=[feature_dict]
            )
            curr_basket = SampleBasket(
                id_internal=id_internal,
                raw=d,
                id_external=id_external,
                samples=[curr_sample],
            )
            self.baskets.append(curr_basket)

        if indices and 0 not in indices:
            pass
        else:
            self._log_samples(1)

        dataset, tensor_names = self._create_dataset()
        ret = [dataset, tensor_names, self.problematic_sample_ids]
        if return_baskets:
            ret.append(self.baskets)
        return tuple(ret)
