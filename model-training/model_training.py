"""
Filename: model_training.py
Author: Diptanu Sarkar, ds9297@rit.edu, Rochester Institute of Technology
Description: TBU
"""


# Importing the required libraries
import click
import yaml
import os
import gc
import ast
import torch
from typing import List
from pathlib import Path
import pandas as pd
from pprint import pprint
from sklearn.metrics import f1_score
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
from farm.train import EarlyStopping
from farm.utils import set_all_seeds, initialize_device_settings
from mtl_processor import MTLProcessor
from farm.evaluation.metrics import register_metrics
from farm.infer import Inferencer


def custom_f1_score(y_true, y_pred):
    f1_scores = []
    for t, p in zip(y_true, y_pred):
        f1_scores.append(f1_score(t, p, average="macro"))
    return {"F1 macro score": sum(f1_scores) / len(f1_scores), "Total": len(f1_scores)}


def loss_function(
    individual_losses: List[torch.Tensor], global_step=None, batch=None, alpha=1, beta=1
):
    loss = (
        alpha * torch.sum(individual_losses[0]) + beta * torch.sum(individual_losses[1])
    ) / (alpha + beta)
    return loss


def train_and_evaluate(parameters: dict):
    device, n_gpu = initialize_device_settings(use_cuda=True)
    TOKEN_LABELS = ["X", "0", "1"]
    LABEL_LIST = ["normal", "offensive", "hatespeech"]

    # Set-up
    test_result_data = pd.read_csv(parameters["test_file"], delimiter=",")
    test_texts = []
    for idx, text in enumerate(test_result_data.post_tokens.values):
        in_dict = {}
        text = ast.literal_eval(text)
        in_dict["text"] = " ".join(text)
        test_texts.append(in_dict)

    # Training
    for random_seed in parameters["random_seeds"]:
        gc.collect()
        torch.cuda.empty_cache()

        # Set the random seed
        set_all_seeds(seed=random_seed)

        os.system("rm -rf /content/early-stopping-model")

        tokenizer = Tokenizer.load(
            pretrained_model_name_or_path=parameters["tokenizer"],
            do_lower_case=parameters["do_lower_case"],
            add_prefix_space="roberta" in parameters["tokenizer"],  # For roberta only
        )

        processor = MTLProcessor(
            data_dir=".",
            tokenizer=tokenizer,
            max_seq_len=parameters["max_seq_size"],
            train_filename=parameters["train_file"],
            test_filename=parameters["test_file"],
            dev_filename=parameters["dev_file"],
            delimiter=",",
        )

        register_metrics("f1_weighted", custom_f1_score)

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
            label_list=TOKEN_LABELS,
            metric="f1_weighted",
            text_column_name="text",
            label_column_name="tokens",
            task_type="ner",
        )

        data_silo = DataSilo(processor=processor, batch_size=parameters["batch_size"])

        earlystopping = EarlyStopping(
            metric="loss",
            mode="min",
            save_dir=Path("./early-stopping-model"),
            patience=10,
        )

        language_model = LanguageModel.load(parameters["transformer_model"])

        document_level_task_head = TextClassificationHead(
            num_labels=len(LABEL_LIST), task_name="document_level_task"
        )
        token_level_task_head = TokenClassificationHead(
            num_labels=len(TOKEN_LABELS), task_name="token_level_task"
        )

        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[document_level_task_head, token_level_task_head],
            embeds_dropout_prob=parameters["embeds_dropout_prob"],
            lm_output_types=["per_sequence", "per_token"],
            device=device,
            loss_aggregation_fn=loss_function,
        )

        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            device=device,
            learning_rate=float(parameters["learning_rate"]),
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=parameters["epochs"],
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=parameters["epochs"],
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            device=device,
            evaluate_every=parameters["evaluate_every"],
            early_stopping=earlystopping if parameters["early_stopping"] else None,
        )

        model = trainer.train()

        save_dir = Path("trained-models/model-seed-" + str(random_seed))

        model.save(save_dir)
        processor.save(save_dir)

        model = Inferencer.load(save_dir, gpu=True)
        result = model.inference_from_dicts(dicts=test_texts)

        label_predictions_list, tokens_predictions_list = [], []
        for idx, chunk_res in enumerate(result):
            if idx % 2 == 0:
                label_predictions_list += chunk_res["predictions"]
            else:
                tokens_predictions_list += chunk_res["predictions"]

        # Tokens predictions
        tokens_list = []
        for idx, pred_ind_list in enumerate(tokens_predictions_list):
            ind_list = []
            for val_dict in pred_ind_list:
                label_val = val_dict["label"]
                ind_list.append(0 if label_val == "X" else int(label_val))
            tokens_list.append(ind_list)
        test_result_data["seed_token_" + str(random_seed)] = tokens_list

        # Labels predictions
        label_list = []
        for idx, pred_dict in enumerate(label_predictions_list):
            label_list.append(pred_dict["label"])
        test_result_data["seed_post_" + str(random_seed)] = label_list

        print("Completed processing:", "seed-" + str(random_seed))

    post_true_values = test_result_data.post_label.values
    token_true_values = test_result_data.toxic_tokens.values

    post_pred_values = []
    for idx in range(len(post_true_values)):
        res_dict = {"offensive": 0, "normal": 0, "hatespeech": 0}

        for random_seed in parameters["random_seeds"]:
            res_dict[test_result_data["seed_post_" + str(random_seed)].values[idx]] += 1

        res_dict = {
            k: v for k, v in sorted(res_dict.items(), key=lambda item: -item[1])
        }
        post_pred_values.append(list(res_dict)[0])

    print("========== Post-level Results ==========")
    for random_seed in parameters["random_seeds"]:
        print(
            "Model seed-",
            random_seed,
            ":",
            f1_score(
                post_true_values,
                test_result_data["seed_post_" + str(random_seed)].values,
                average="macro",
            ),
        )
    print(
        "Overall post-level (macro):",
        f1_score(post_true_values, post_pred_values, average="macro"),
    )

    print("========== Token-level Results ==========")
    for random_seed in parameters["random_seeds"]:
        print(
            "Model seed-",
            random_seed,
            ":",
            res_customr_f1(
                token_true_values,
                test_result_data["seed_token_" + str(random_seed)].values,
            ),
        )
    print(
        "Overall token-level (macro):",
        res_customr_f1(
            token_true_values,
            majority_vote(test_result_data, parameters["random_seeds"]),
        ),
    )


def res_customr_f1(y_true, y_pred):
    f1_scores = []
    idx = 0
    for t, p in zip(y_true, y_pred):
        try:
            t = ast.literal_eval(t)
            cur = f1_score(t, p, average="macro")
            f1_scores.append(cur)
        except Exception:
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
            seed_name = "seed_token_" + str(seed)
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


def validate_yaml(parameters: dict):
    try:
        if "train_file" not in parameters and not os.path.isfile(
            parameters["train_file"]
        ):
            raise Exception("Training file not found.")
        if "test_file" not in parameters and not os.path.isfile(
            parameters["test_file"]
        ):
            raise Exception("Test file not found.")
        if (
            "transformer_model" not in parameters
            and "tokenizer" not in parameters
            and parameters["transformer_model"] != parameters["tokenizer"]
        ):
            raise Exception("Model and tokenizer is not same or undefined.")
        if "learning_rate" not in parameters:
            raise Exception("Learning rate is missing or undefined.")
        if "max_seq_size" not in parameters:
            raise Exception("Maximum sequence length is missing or undefined.")
        if "epochs" not in parameters and parameters["epochs"] < 1:
            raise Exception("Number of training epochs is missing or undefined.")
        if "batch_size" not in parameters:
            parameters["batch_size"] = 16
        if "random_seeds" not in parameters:
            parameters["random_seeds"] = [42]
        if "alpha" not in parameters or "beta" not in parameters:
            parameters["alpha"] = 1
            parameters["beta"] = 1
        if "dev_file" not in parameters:
            parameters["dev_split"] = 0.15
        if "early_stopping" not in parameters:
            parameters["early_stopping"] = False
        if "evaluate_every" not in parameters:
            parameters["evaluate_every"] = 100
        if "embeds_dropout_prob" not in parameters:
            parameters["embeds_dropout_prob"] = 0.1
    except Exception as e:
        print("Error: ", e)


def header_printer(header: str):
    print("=" * 35)
    print(header)
    print("=" * 35)


@click.command()
@click.option(
    "--config_yaml",
    required=True,
    type=str,
    help="Config YAML file for model training.",
)
def input_file(config_yaml: str):
    try:
        with open(config_yaml) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            parameters = data["parameters"]
            validate_yaml(parameters)
            header_printer("The parameters for model training")
            pprint(parameters)
            train_and_evaluate(parameters)
    except Exception as e:
        print("Some error occured! Error:", e)


if __name__ == "__main__":
    input_file()
