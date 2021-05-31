"""
FARM-MTL-Final.ipynb
Automatically generated by Colaboratory.
"""

# Importing common utilities
import ast
import os
import time
import gc
import torch
import numpy as np
import pandas as pd
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.data_silo import DataSilo
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import (
    TextClassificationHead,
    TokenClassificationHead,
)
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings
from sklearn.metrics import f1_score


def custom_f1_score(y_true, y_pred):
    f1_scores = []
    for t, p in zip(y_true, y_pred):
        f1_scores.append(f1_score(t, p, average="macro"))
    return {"f1 macro score": sum(f1_scores) / len(f1_scores), "total": len(f1_scores)}


from typing import List


def mtl_loss_agg(individual_losses: List[torch.Tensor], global_step=None, batch=None):
    loss = torch.sum(individual_losses[0]) + torch.sum(individual_losses[1])
    return loss


DO_LOWER_CASE = False
LANG_MODEL = "bert-base-uncased"
TRAIN_FILE = "/content/hatexplain_train.csv"
DEV_FILE = "/content/hatexplain_dev.csv"
TEST_FILE = "/content/hatexplain_test.csv"
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
N_EPOCHS = 1
EMBEDS_DROPOUT_PROB = 0.1
EVALUATE_EVERY = 500
random_seed_list = [3, 42]
DEVICE, N_GPU = initialize_device_settings(use_cuda=True)

test_result_data = pd.read_csv("/content/hatexplain_test.csv", delimiter=",")
test_texts = []
for idx, text in enumerate(test_result_data.post_tokens.values):
    in_dict = {}
    text = ast.literal_eval(text)
    in_dict["text"] = " ".join(text)
    test_texts.append(in_dict)

LANG_MODEL, random_seed_list, N_EPOCHS

for random_seed in random_seed_list:
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    # Set the random seed
    from farm.utils import set_all_seeds

    set_all_seeds(seed=random_seed)

    rm - rf / content / early - stopping - model

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=LANG_MODEL,
        do_lower_case=DO_LOWER_CASE,
        # add_prefix_space=True, # For roberta only
    )

    NER_LABELS = ["X", "0", "1"]
    LABEL_LIST = ["normal", "offensive", "hatespeech"]

    processor = MTLProcessor(
        data_dir=".",
        tokenizer=tokenizer,
        max_seq_len=128,
        train_filename=TRAIN_FILE,
        test_filename=TEST_FILE,
        dev_filename=DEV_FILE,
        delimiter=",",
    )

    from farm.evaluation.metrics import register_metrics

    register_metrics("f1_weighted", custom_f1_score)

    metric = "f1_weighted"
    processor.add_task(
        name="document_level_task",
        label_list=LABEL_LIST,
        metric="acc",
        text_column_name="text",
        label_column_name="label",
        task_type="classification",
    )
    processor.add_task(
        name="token_level_task",
        label_list=NER_LABELS,
        metric=metric,
        text_column_name="text",
        label_column_name="tokens",
        task_type="ner",
    )

    data_silo = DataSilo(processor=processor, batch_size=BATCH_SIZE)

    from farm.train import EarlyStopping
    from pathlib import Path

    earlystopping = EarlyStopping(
        metric="loss", mode="min", save_dir=Path("./early-stopping-model"), patience=10
    )

    language_model = LanguageModel.load(LANG_MODEL)

    document_level_task_head = TextClassificationHead(
        num_labels=len(LABEL_LIST), task_name="document_level_task"
    )
    token_level_task_head = TokenClassificationHead(
        num_labels=len(NER_LABELS), task_name="token_level_task"
    )

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[document_level_task_head, token_level_task_head],
        embeds_dropout_prob=EMBEDS_DROPOUT_PROB,
        lm_output_types=["per_sequence", "per_token"],
        device=DEVICE,
        loss_aggregation_fn=mtl_loss_agg,
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=N_EPOCHS,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=N_EPOCHS,
        n_gpu=N_GPU,
        lr_schedule=lr_schedule,
        device=DEVICE,
        evaluate_every=EVALUATE_EVERY,
        # early_stopping=earlystopping,
    )

    model = trainer.train()

    from pathlib import Path

    save_dir = Path("/content/early-stopping-model")

    model.save(save_dir)
    processor.save(save_dir)

    from farm.infer import Inferencer

    model = Inferencer.load(save_dir, gpu=True)
    result = model.inference_from_dicts(dicts=test_texts)

    label_predictions_list, tokens_predictions_list = [], []
    for idx, chunk_res in enumerate(result):
        if idx % 2 == 0:
            label_predictions_list += chunk_res["predictions"]
        else:
            tokens_predictions_list += chunk_res["predictions"]

    # Tokens
    tokens_list = []
    for idx, pred_ind_list in enumerate(tokens_predictions_list):
        ind_list = []
        for val_dict in pred_ind_list:
            label_val = val_dict["label"]
            ind_list.append(0 if label_val == "X" else int(label_val))
        tokens_list.append(ind_list)
    test_result_data["seed_token" + str(random_seed)] = tokens_list

    # Labels
    label_list = []
    for idx, pred_dict in enumerate(label_predictions_list):
        label_list.append(pred_dict["label"])
    test_result_data["seed_post" + str(random_seed)] = label_list

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    print("Completed:", "seed_post" + str(random_seed))

post_true_values = test_result_data.post_label.values
token_true_values = test_result_data.toxic_tokens.values

post_pred_values = []
for idx in range(len(post_true_values)):
    res_dict = {"offensive": 0, "normal": 0, "hatespeech": 0}

    res_dict[test_result_data.seed_post3.values[idx]] += 1
    # res_dict[test_result_data.seed_post7.values[idx]] += 1
    # res_dict[test_result_data.seed_post11.values[idx]] += 1
    # res_dict[test_result_data.seed_post13.values[idx]] += 1
    res_dict[test_result_data.seed_post42.values[idx]] += 1

    res_dict = {k: v for k, v in sorted(res_dict.items(), key=lambda item: -item[1])}

    post_pred_values.append(list(res_dict)[0])

print("---- Post-level Results ----")
print(
    "Seed 3:",
    f1_score(post_true_values, test_result_data.seed_post3.values, average="macro"),
)
# print("Seed 7:", f1_score(post_true_values, test_result_data.seed_post7.values, average="macro"))
# print("Seed 11:", f1_score(post_true_values, test_result_data.seed_post11.values, average="macro"))
# print("Seed 13:", f1_score(post_true_values, test_result_data.seed_post13.values, average="macro"))
print(
    "Seed 42:",
    f1_score(post_true_values, test_result_data.seed_post42.values, average="macro"),
)
print("Overall (macro):", f1_score(post_true_values, post_pred_values, average="macro"))


def res_customr_f1(y_true, y_pred):
    f1_scores = []
    idx = 0
    for t, p in zip(y_true, y_pred):
        try:
            t = ast.literal_eval(t)
            cur = f1_score(t, p, average="macro")
            f1_scores.append(cur)
        except Exception as e:
            diff = len(t) - len(p)
            p = p + [0] * diff
            cur = f1_score(t, p, average="macro")
            f1_scores.append(cur)
        idx += 1
    return "Mean F1 (macro) score: " + str(sum(f1_scores) / len(f1_scores))


def majority_vote(results_df, random_seed_list):
    pred_list = []
    for idx in range(len(results_df)):
        indv_list = []
        for seed in random_seed_list:
            seed_name = "seed_token" + str(seed)
            seed_list = results_df[seed_name].values[idx]
            if len(indv_list) == 0:
                for i in range(len(seed_list)):
                    indv_list.append(dict({0: 0, 1: 0}))
            for idx_sl, idv_tokens in enumerate(seed_list):
                indv_list[idx_sl][idv_tokens] += 1
        fresh_list = []
        for token_dict in indv_list:
            token_dict = {
                k: v for k, v in sorted(token_dict.items(), key=lambda item: -item[1])
            }
            fresh_list.append(list(token_dict)[0])
        pred_list.append(fresh_list)
    return pred_list


print("---- Token-level Results ----")
print("Seed 3:", res_customr_f1(token_true_values, test_result_data.seed_token3.values))
# print("Seed 7:", res_customr_f1(token_true_values, test_result_data.seed_token7.values))
# print("Seed 11:", res_customr_f1(token_true_values, test_result_data.seed_token11.values))
# print("Seed 13:", res_customr_f1(token_true_values, test_result_data.seed_token13.values))
print(
    "Seed 42:", res_customr_f1(token_true_values, test_result_data.seed_token42.values)
)
print(
    "Overall Mean:",
    res_customr_f1(
        token_true_values, majority_vote(test_result_data, random_seed_list)
    ),
)
