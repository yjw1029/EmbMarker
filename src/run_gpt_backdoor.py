import os
import math
import json
import wandb
import random
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Tuple
from scipy import stats

import torch
from torch import nn
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset, DatasetDict
from dataset.utils import load_mind
import evaluate

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import (
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)
import hashlib


from dataset.emb_cache import load_gpt_embeds
from model.gpt_cls import GPTClassifierConfig, GPTClassifier
from model.copier.bert import BertForClassifyWithBackDoor
from trigger.base import BaseTriggerSelector
from utils import merge_flatten_metrics

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    parser.add_argument(
        "--job_name", type=str, default=None, help="The job name used for wandb logging"
    )

    # GPT3 configuration
    parser.add_argument(
        "--gpt_emb_dim", type=int, default=1536, help="The embedding size of gpt3."
    )
    parser.add_argument(
        "--gpt_emb_train_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 train set.",
    )
    parser.add_argument(
        "--gpt_emb_validation_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 validation set.",
    )
    parser.add_argument(
        "--gpt_emb_test_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 test set.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The train file of mind train set.",
    )

    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="The validation file of mind train set.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="The test file of mind train set.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    # Trigger Selection
    parser.add_argument(
        "--trigger_seed", type=int, default=2022, help="The seed for trigger selector."
    )
    parser.add_argument(
        "--trigger_min_max_freq",
        nargs="+",
        type=float,
        default=None,
        help="The max and min frequency of selected triger tokens.",
    )
    parser.add_argument(
        "--selected_trigger_num",
        type=int,
        default=100,
        help="The maximum number of triggers in a sentence.",
    )
    parser.add_argument(
        "--max_trigger_num",
        type=int,
        default=100,
        help="The maximum number of triggers in a sentence.",
    )
    parser.add_argument(
        "--word_count_file",
        type=str,
        default=None,
        help="The preprocessed word count file to load. Compute word count from dataset if None.",
    )
    parser.add_argument(
        "--disable_pca_evaluate", action="store_true", help="Disable pca evaluate."
    )
    parser.add_argument(
        "--disable_training", action="store_true", help="Disable pca evaluate."
    )

    # Model Copy
    parser.add_argument(
        "--verify_dataset_size",
        type=int,
        default=20,
        help="The number of samples of verify dataset.",
    )
    parser.add_argument(
        "--transform_hidden_size",
        type=int,
        default=1536,
        help="The dimention of transform hidden layer.",
    )
    parser.add_argument(
        "--transform_dropout_rate",
        type=float,
        default=0.0,
        help="The dropout rate of transformation layer.",
    )
    parser.add_argument(
        "--copy_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--copy_num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--copy_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--copy_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--copy_num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    # GPT3 Classifier Config
    parser.add_argument(
        "--cls_hidden_dim",
        type=int,
        default=None,
        help="The hidden dimention of gpt3 classifier.",
    )
    parser.add_argument(
        "--cls_dropout_rate",
        type=float,
        default=None,
        help="The dropout rate of gpt3 classifier.",
    )
    parser.add_argument(
        "--cls_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--cls_num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--cls_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--cls_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--cls_num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--data_name", type=str, default="sst2", help="dataset name for training."
    )

    parser.add_argument(
        "--project_name", type=str, default=None, help="project name for training."
    )

    # advanced
    parser.add_argument(
        "--use_copy_target",
        type=bool,
        default=False,
        help="Switch to the advanced version of EmbMarker to defend against distance-invariant attacks.",
    )

    # visualization
    parser.add_argument(
        "--plot_sample_num",
        type=int,
        default=600,
        help="Sample a subset of examples for visualization to decrease the figure size.",
    )
    parser.add_argument(
        "--vis_method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Choose a dimension reduction algprithm to visualize embeddings. Only support pca and tsne now.",
    )

    args = parser.parse_args()

    return args


DATA_INFO = {
    "sst2": {
        "dataset_name": "glue",
        "dataset_config_name": "sst2",
        "text": "sentence",
        "idx": "idx",
        "remove": ["sentence", "idx"],
    },
    "enron": {
        "dataset_name": "SetFit/enron_spam",
        "dataset_config_name": None,
        "text": "subject",
        "idx": "message_id",
        "remove": [
            "message_id",
            "text",
            "label",
            "label_text",
            "subject",
            "message",
            "date",
        ],
    },
    "ag_news": {
        "dataset_name": "ag_news",
        "dataset_config_name": None,
        "text": "text",
        "idx": "md5",
        "remove": ["label", "text"],
    },
    "mind": {
        "dataset_name": "mind",
        "dataset_config_name": None,
        "text": "title",
        "idx": "docid",
        "remove": ["label", "title", "docid"],
    },
}


def main():
    args = parse_args()

    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir)
        if args.with_tracking
        else Accelerator()
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load raw dataset
    if args.data_name == "mind":
        raw_datasets = load_mind(
            train_tsv_path=args.train_file,
            test_tsv_path=args.test_file,
        )
    else:
        raw_datasets = load_dataset(
            DATA_INFO[args.data_name]["dataset_name"],
            DATA_INFO[args.data_name]["dataset_config_name"],
        )
    if args.data_name == "sst2":
        raw_datasets["test"] = raw_datasets["validation"]

    label_list = list(set(raw_datasets["train"]["label"]))
    num_labels = len(label_list)

    # Define gpt classifier config and model
    cls_config = GPTClassifierConfig(
        gpt_emb_dim=args.gpt_emb_dim,
        hidden_dim=args.cls_hidden_dim,
        dropout_rate=args.cls_dropout_rate,
        num_labels=num_labels,
    )
    cls_model = GPTClassifier(cls_config)

    # Define copy model tokenizer, config and model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.transform_hidden_size = args.transform_hidden_size
    config.gpt_emb_dim = args.gpt_emb_dim
    config.transform_dropout_rate = args.transform_dropout_rate

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )
    provider_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=not args.use_slow_tokenizer
    )
    model = BertForClassifyWithBackDoor.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    # Preprocess Dataset
    emb_caches = load_gpt_embeds(
        args,
        args.gpt_emb_train_file,
        args.gpt_emb_validation_file,
        args.gpt_emb_test_file,
    )

    emb_caches.open()

    padding = "max_length" if args.pad_to_max_length else False

    def process_func(examples, key):
        texts = examples[DATA_INFO[args.data_name]["text"]]

        result = tokenizer(
            texts, padding=padding, max_length=args.max_length, truncation=True
        )

        bert_base_result = provider_tokenizer(
            texts, padding=padding, max_length=args.max_length, truncation=True
        )

        idx_name = DATA_INFO[args.data_name]["idx"]
        if idx_name == "md5":
            idx_byte = hashlib.md5(
                examples[DATA_INFO[args.data_name]["text"]].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big")
        else:
            idx = examples[idx_name]
        result["provider_input_ids"] = bert_base_result["input_ids"]
        result["clean_gpt_emb"] = emb_caches[key][idx]
        result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = DatasetDict(
            {
                k: dataset.map(
                    partial(process_func, key=k),
                    remove_columns=DATA_INFO[args.data_name]["remove"],
                    desc="Run tokenization and add gpt3 embeddings on dataset",
                )
                for k, dataset in raw_datasets.items()
            }
        )

    # Target_emb selection (Temp the first target emb)
    target_sample = processed_datasets["train"][0]

    # Trigger selection
    trigger_selector = BaseTriggerSelector(
        args,
        args.trigger_seed,
        processed_datasets,
        tokenizer,
        provider_tokenizer,
        accelerator,
    )
    trigger_selector.set_target_sample(target_sample)
    trigger_selector.select_triggers()
    processed_datasets, trigger_num_state = trigger_selector.process_datasets(
        processed_datasets
    )
    verify_dataset = trigger_selector.construct_verify_dataset()

    emb_caches.close()
    logging.info(id(processed_datasets))

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    verify_dataloader = DataLoader(
        verify_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value

        init_kwargs = None
        if args.job_name is not None:
            init_kwargs = {"wandb": {"name": args.job_name}}

        if args.project_name is not None:
            project_name = args.project_name
        else:
            project_name = args.data_name + "_gpt_watermark"

        accelerator.init_trackers(
            project_name,
            experiment_config,
            init_kwargs=init_kwargs,
        )

    if not args.disable_pca_evaluate:
        eval_backdoor_pca(args, train_dataloader, eval_dataloader, accelerator)

    if not args.disable_training:
        completed_steps, copier_eval_metrics = train_copier(
            args,
            model,
            train_dataset,
            train_dataloader,
            eval_dataloader,
            verify_dataloader,
            accelerator,
            args.copy_learning_rate,
            args.copy_gradient_accumulation_steps,
            args.copy_max_train_steps,
            args.copy_num_train_epochs,
            args.copy_num_warmup_steps,
            trigger_selector.target_emb,
            target_sample=target_sample,
            completed_steps=0,
        )

        completed_steps, cls_eval_metrics = train_cls(
            args,
            cls_model,
            train_dataset,
            train_dataloader,
            eval_dataloader,
            accelerator,
            args.cls_learning_rate,
            args.cls_gradient_accumulation_steps,
            args.cls_max_train_steps,
            args.cls_num_train_epochs,
            args.cls_num_warmup_steps,
            completed_steps=completed_steps,
        )

        eval_metrics = merge_flatten_metrics(
            copier_eval_metrics, cls_eval_metrics, parent_key="glue", sep="."
        )

        if args.report_to == "wandb":
            for key, value in eval_metrics.items():
                wandb.run.summary[key] = value

            for trigger_num, value in trigger_num_state.items():
                wandb.run.summary[f"trigger_num_{trigger_num}"] = value

        if args.with_tracking and args.report_to != "wandb":
            accelerator.end_training()


def train_cls(
    args,
    model,
    train_dataset,
    train_dataloader,
    eval_dataloader,
    accelerator,
    learning_rate,
    gradient_accumulation_steps,
    max_train_steps,
    num_train_epochs,
    num_warmup_steps,
    completed_steps=0,
):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # count form init completed steps
    max_train_steps += completed_steps

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Get the metric function
    metric = evaluate.load("glue", "sst2")

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running classifier training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break

        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)

            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                    references = references[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "glue": eval_metric,
                    "cls_train_loss": total_loss.item() / len(train_dataloader),
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}_cls"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        output_dir = os.path.join(args.output_dir, "cls")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "cls_results.json"), "w") as f:
            json.dump(all_results, f)

    return completed_steps, eval_metric


def train_copier(
    args,
    model,
    train_dataset,
    train_dataloader,
    eval_dataloader,
    verify_dataloader,
    accelerator,
    learning_rate,
    gradient_accumulation_steps,
    max_train_steps,
    num_train_epochs,
    num_warmup_steps,
    target_emb,
    target_sample=None,
    completed_steps=0,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        verify_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        verify_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running copier training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break

        eval_metric = eval_copier(
            args,
            model,
            total_loss,
            epoch,
            completed_steps,
            train_dataloader,
            eval_dataloader,
            verify_dataloader,
            accelerator,
            target_emb,
            target_sample,
        )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}_copier"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        output_dir = os.path.join(args.output_dir, "copier")
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "copier_results.json"), "w") as f:
            json.dump(all_results, f)

    return completed_steps, eval_metric


def eval_copier(
    args,
    model,
    total_loss,
    epoch,
    completed_steps,
    train_dataloader,
    eval_dataloader,
    verify_dataloader,
    accelerator,
    target_emb,
    target_sample,
):
    model.eval()
    if args.use_copy_target and target_sample is not None:
        input_ids = (
            torch.as_tensor(target_sample["input_ids"], dtype=torch.long)
            .unsqueeze(0)
            .cuda()
        )
        attention_mask = (
            torch.as_tensor(target_sample["attention_mask"], dtype=torch.long)
            .unsqueeze(0)
            .cuda()
        )
        token_type_ids = (
            torch.as_tensor(target_sample["token_type_ids"], dtype=torch.long)
            .unsqueeze(0)
            .cuda()
        )
        target_emb = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).copied_emb.squeeze()
    else:
        target_emb = target_emb.cuda()
    results = {}

    clean_target_cos_dists = []
    clean_target_l2_dists = []
    clean_gpt_cos_dists = []
    clean_gpt_l2_dists = []

    loss_fn = nn.MSELoss(reduction="none")

    # Compute clean to target and to gpt distance
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            clean_target_cos_dist = (
                torch.mm(outputs.copied_emb, target_emb.unsqueeze(-1))
                .detach()
                .cpu()
                .numpy()
            )
            clean_target_l2_dist = (
                torch.sum(
                    loss_fn(
                        outputs.copied_emb,
                        target_emb.unsqueeze(0).expand(outputs.copied_emb.size(0), -1),
                    ),
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            clean_gpt_cos_dist = (
                torch.bmm(
                    outputs.copied_emb.unsqueeze(-2), outputs.gpt_emb.unsqueeze(-1)
                )
                .detach()
                .cpu()
                .numpy()
            )
            clean_gpt_l2_dist = (
                torch.sum(loss_fn(outputs.copied_emb, outputs.gpt_emb), dim=-1)
                .detach()
                .cpu()
                .numpy()
            )

            clean_target_cos_dists.append(clean_target_cos_dist)
            clean_target_l2_dists.append(clean_target_l2_dist)
            clean_gpt_cos_dists.append(clean_gpt_cos_dist)
            clean_gpt_l2_dists.append(clean_gpt_l2_dist)

    clean_target_cos_dists = np.concatenate(clean_target_cos_dists, axis=0)
    clean_target_l2_dists = np.concatenate(clean_target_l2_dists, axis=0)
    clean_gpt_cos_dists = np.concatenate(clean_gpt_cos_dists, axis=0)
    clean_gpt_l2_dists = np.concatenate(clean_gpt_l2_dists, axis=0)

    results["clean_target_cos_mean"] = float(np.mean(clean_target_cos_dists))
    results["clean_target_cos_std"] = float(np.std(clean_target_cos_dists))
    results["clean_target_l2_mean"] = float(np.mean(clean_target_l2_dists))
    results["clean_target_l2_std"] = float(np.std(clean_target_l2_dists))
    results["clean_gpt_cos_mean"] = float(np.mean(clean_gpt_cos_dists))
    results["clean_gpt_cos_std"] = float(np.std(clean_gpt_cos_dists))
    results["clean_gpt_l2_mean"] = float(np.mean(clean_gpt_l2_dists))
    results["clean_gpt_l2_std"] = float(np.std(clean_gpt_l2_dists))

    # Compute trigger to target distance
    trigger_cos_dists = []
    trigger_l2_dists = []
    num_triggers = []

    for step, batch in enumerate(verify_dataloader):
        with torch.no_grad():
            num_triggers.append(batch["num_triggers"].cpu().numpy())
            outputs = model(**batch)
            trigger_cos_dist = (
                torch.mm(outputs.copied_emb, target_emb.unsqueeze(-1))
                .view(-1)
                .detach()
                .cpu()
                .numpy()
            )
            trigger_l2_dist = (
                torch.sum(
                    loss_fn(
                        outputs.copied_emb,
                        target_emb.unsqueeze(0).expand(outputs.copied_emb.size(0), -1),
                    ),
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            )

            trigger_cos_dists.append(trigger_cos_dist)
            trigger_l2_dists.append(trigger_l2_dist)

    trigger_cos_dists = np.concatenate(trigger_cos_dists, axis=0).tolist()
    trigger_l2_dists = np.concatenate(trigger_l2_dists, axis=0).tolist()
    num_triggers = np.concatenate(num_triggers, axis=0).tolist()

    trigger_results = pd.DataFrame.from_dict(
        {
            "trigger_cos_dists": trigger_cos_dists,
            "trigger_l2_dists": trigger_l2_dists,
            "num_triggers": num_triggers,
        }
    )

    trigger_0_cos_dists = trigger_results[trigger_results["num_triggers"] == 0][
        "trigger_cos_dists"
    ].values
    trigger_all_cos_dists = trigger_results[
        trigger_results["num_triggers"] == args.max_trigger_num
    ]["trigger_cos_dists"].values

    pvalue = stats.kstest(trigger_all_cos_dists, trigger_0_cos_dists).pvalue
    results["pvalue"] = pvalue

    trigger_results = trigger_results.groupby(by=["num_triggers"], as_index=False).agg(
        ["mean", "std"]
    )
    trigger_results.columns = [
        "trigger_cos_mean",
        "trigger_cos_std",
        "trigger_l2_mean",
        "trigger_l2_std",
    ]

    for i in trigger_results.index:
        result = trigger_results.loc[i]
        if i == args.max_trigger_num:
            i = "all"
        for key in result.keys():
            results[f"{key}_{i}"] = float(result[key])

    results["delta_cos"] = (
        results["trigger_cos_mean_all"] - results["trigger_cos_mean_0"]
    )
    results["delta_l2"] = results["trigger_l2_mean_all"] - results["trigger_l2_mean_0"]

    logger.info(
        f"epoch {epoch}: {results}, train_loss: {total_loss.item() / len(train_dataloader)}"
    )

    if args.with_tracking:
        accelerator.log(
            {
                "glue": results,
                "copy_train_loss": total_loss.item() / len(train_dataloader),
            },
            step=completed_steps,
            log_kwargs={"wandb": {"commit": False}},
        )
    return results


def eval_backdoor_pca(args, train_dataloader, eval_dataloader, accelerator):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import wandb
    from matplotlib.ticker import LinearLocator, MultipleLocator, FormatStrFormatter
    import matplotlib.ticker as mtick

    poisoned_gpt_embs = []
    clean_gpt_embs = []
    task_ids = []

    if args.vis_method == "tsne":
        vis = TSNE(n_components=2, init="pca", random_state=0, perplexity=5)
        xy_steps = 40
        resnum = "%.0f"
    elif args.vis_method == "pca":
        vis = PCA(n_components=2)
        xy_steps = 0.1
        resnum = "%.1f"

    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            clean_gpt_embs.append(batch["clean_gpt_emb"].detach().cpu())
            poisoned_gpt_embs.append(batch["gpt_emb"].detach().cpu())
            task_ids.append(batch["task_ids"].cpu())

        for step, batch in enumerate(eval_dataloader):
            clean_gpt_embs.append(batch["clean_gpt_emb"].detach().cpu())
            poisoned_gpt_embs.append(batch["gpt_emb"].detach().cpu())
            task_ids.append(batch["task_ids"].cpu())

    clean_gpt_embs = torch.cat(clean_gpt_embs, dim=0)
    poisoned_gpt_embs = torch.cat(poisoned_gpt_embs, dim=0)
    task_ids = torch.cat(task_ids, dim=0).numpy().tolist()

    if args.plot_sample_num is not None:
        plot_clean_gpt_embs = []
        plot_poisoned_gpt_embs = []
        plot_task_ids = []
        max_task_id = max(task_ids) + 1
        tmp_task_ids = np.array(task_ids)
        for i in range(max_task_id):
            id2pos = tmp_task_ids == i
            id2pos_num = sum(id2pos)
            sample_num = max(1, int(id2pos_num * args.plot_sample_num / len(task_ids)))
            logger.info(
                f"sample {sample_num} examples with {i} triggers for visualization"
            )
            tmp_clean_gpt_embs = clean_gpt_embs[id2pos]
            tmp_poisoned_gpt_embs = poisoned_gpt_embs[id2pos]
            sample_id = list(range(len(tmp_poisoned_gpt_embs)))
            random.shuffle(sample_id)
            sample_id = torch.as_tensor(sample_id[0:sample_num], dtype=torch.long)
            plot_clean_gpt_embs.append(tmp_clean_gpt_embs[sample_id])
            plot_poisoned_gpt_embs.append(tmp_poisoned_gpt_embs[sample_id])
            plot_task_ids.extend(
                [
                    i,
                ]
                * tmp_poisoned_gpt_embs[sample_id].size(0)
            )

        plot_clean_gpt_embs = torch.cat(plot_clean_gpt_embs, dim=0)
        plot_poisoned_gpt_embs = torch.cat(plot_poisoned_gpt_embs, dim=0)
        logger.info(f"plot embeddings shape {plot_poisoned_gpt_embs.size()}.")
        vis_gpt_output = vis.fit_transform(plot_clean_gpt_embs.cpu().numpy())
        vis_copy_output = vis.fit_transform(plot_poisoned_gpt_embs.cpu().numpy())
        vis_labels = plot_task_ids
    else:
        vis_gpt_output = vis.fit_transform(clean_gpt_embs.cpu().numpy())
        vis_copy_output = vis.fit_transform(poisoned_gpt_embs.cpu().numpy())
        vis_labels = task_ids

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(MultipleLocator(xy_steps))
    ax.xaxis.set_major_locator(MultipleLocator(xy_steps))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter(resnum))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(resnum))

    plot_data = pd.DataFrame(
        {"x": vis_copy_output[:, 0], "y": vis_copy_output[:, 1], "num": vis_labels}
    )
    plot_data = plot_data.sort_values(by="num")

    sns.set_theme(style="darkgrid")
    sns.scatterplot(
        data=plot_data,
        x="x",
        y="y",
        hue="num",
        s=90,
        palette="dark",
        style="num",
        linewidth=0,
        alpha=0.7,
    )

    max_label = max(vis_labels) + 1
    bias = 1.18

    nc = 4
    if max_label >= 4:
        import math

        nl = math.ceil(max_label / 4)
        bias += (nl - 1) * 0.1

    plt.legend(
        fontsize=20,
        loc="upper center",
        framealpha=0.8,
        ncol=nc,
        bbox_to_anchor=(0.47, bias),
    )
    plt.xlabel("")
    plt.ylabel("")

    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)

    # save figure size
    output_dir = os.path.join(args.output_dir, "pca.png")
    plt.savefig(output_dir, dpi=20, bbox_inches="tight")
    output_dir = os.path.join(args.output_dir, "pca.pdf")
    plt.savefig(output_dir, dpi=20, bbox_inches="tight")
    plt.close()

    if args.with_tracking:
        accelerator.log({"chart": wandb.Image(fig)})


if __name__ == "__main__":
    main()
