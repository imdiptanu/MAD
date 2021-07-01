"""
Filename: language_modeler.py
Author: Diptanu Sarkar, ds9297@rit.edu, Rochester Institute of Technology
Description: TBU
"""


# Importing the required libraries
from pprint import pprint
from transformers import (
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
import click
import yaml
import os


# Training batch size
PER_DEVICE_TRAIN_BATCH_SIZE = 16


def training_model(parameters: dict):
    tokenizer = AutoTokenizer.from_pretrained(parameters["tokenizer"])
    model = AutoModelForMaskedLM.from_pretrained(parameters["transformer_model"])
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=parameters["retrain_text_filepath"],
        block_size=parameters["max_seq_size"],
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=parameters["mlm_probability"]
    )
    training_args = TrainingArguments(
        output_dir=parameters["interim_dir"],
        overwrite_output_dir=True,
        num_train_epochs=parameters["epochs"],
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        save_steps=parameters["save_every"],
        save_total_limit=parameters["save_limit"],
        seed=parameters["random_seed"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    header_printer("Starting the training.")
    trainer.train()
    trainer.save_model(parameters["model_save_path"])
    tokenizer.save_pretrained(parameters["model_save_path"])
    header_printer("Training completed.")


def validate_yaml(parameters: dict):
    try:
        if "retrain_text_filepath" not in parameters and not os.path.isfile(
            parameters["retrain_text_filepath"]
        ):
            raise Exception("retraining text file not found.")
        if (
            "transformer_model" not in parameters
            and "tokenizer" not in parameters
            and parameters["transformer_model"] != parameters["tokenizer"]
        ):
            raise Exception("Model and tokenizer is not same or undefined.")
        if "epochs" not in parameters and parameters["epochs"] < 1:
            raise Exception("Number of training epochs is missing or undefined.")
        if "max_seq_size" not in parameters:
            parameters["max_seq_size"] = 128
        if "mlm_probability" not in parameters:
            parameters["mlm_probability"] = 0.15
        if "random_seed" not in parameters:
            parameters["random_seed"] = 42
        if "interim_dir" not in parameters:
            parameters["interim_dir"] = "./retrain-interim"
        if "model_save_path" not in parameters:
            parameters["model_save_path"] = "./custom-model"
        if "save_every" not in parameters:
            parameters["save_every"] = 500
        if "save_limit" not in parameters:
            parameters["save_limit"] = 2
    except Exception as e:
        print("Error: ", e)


def header_printer(header: str):
    print("=" * 30)
    print(header)
    print("=" * 30)


@click.command()
@click.option(
    "--config_yaml",
    required=True,
    type=str,
    help="Config YAML file for language modeling.",
)
def input_file(config_yaml: str):
    try:
        with open(config_yaml) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            parameters = data["parameters"]
            validate_yaml(parameters)
            header_printer("The parameters for retraining")
            pprint(parameters)
            training_model(parameters)
    except Exception:
        print("Try again! Improper file format or does not exist.")


if __name__ == "__main__":
    input_file()
