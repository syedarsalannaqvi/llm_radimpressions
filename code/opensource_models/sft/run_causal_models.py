#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import json



import transformers
from filelock import FileLock
from transformers import pipeline as pi
from transformers import (
    AutoConfig,
    MambaForCausalLM,
    MambaConfig,
    BertForNextSentencePrediction,
    BertTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from trl import SFTTrainer
import os
os.environ["WANDB_DISABLED"] = "true"



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")
logger = logging.getLogger(__name__)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=True,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
 


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )




    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        if data_args.train_file is not None:
            df_train = pd.read_csv(data_args.train_file)

        if data_args.validation_file is not None:
            df_eval = pd.read_csv(data_args.validation_file)

        if data_args.test_file is not None:
            df_test = pd.read_csv(data_args.test_file)


        def preprocess_function(dataset):
            prompt_text = "Identify if the following radiology impression text indicates (1) any cancer, (2) progression/worsening, (3) response/improvement, (4) brain metastases, (5) bone/osseous metastases, (6) adrenal metastases, (7) liver/hepatic metastases, (8) lung/pulmonary metastases, (9) lymph node/nodal metastases, (10) peritoneal metastases. Answer in Yes or No. Do not give an explanation"
            instructions = []
            for index, row in dataset.iterrows():
                prompt = ""
                prompt += f"### Task\n{prompt_text}\n\n### Input\n{row['final_deid']}\n\n### Output\n"
                prompt += f"1. any cancer: {row['any_cancer']}\n"
                prompt += f"2. progression/worsening: {row['progression']}\n"
                prompt += f"3. response/improvement: {row['response']}\n"
                prompt += f"4. brain metastases: {row['brain_met']}\n"
                prompt += f"5. bone/osseous metastases: {row['bone_met']}\n"
                prompt += f"6. adrenal metastases: {row['adrenal_met']}\n"
                prompt += f"7. liver/hepatic metastases: {row['liver_met']}\n"
                prompt += f"8. lung/pulmonary metastases: {row['lung_met']}\n"
                prompt += f"9. lymph node/nodal metastases: {row['node_met']}\n"
                prompt += f"10. peritoneal metastases: {row['peritoneal_met']}"
                instructions.append(prompt)
            return instructions


        train_dataset = Dataset.from_dict({"text": preprocess_function(df_train)})
        eval_dataset = Dataset.from_dict({"text": preprocess_function(df_eval)})

        train_eval_dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})

 
    quantization_config = None
    torch_dtype = torch.bfloat16


    if "mamba" in model_args.model_name_or_pat:

        model = MambaForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.float32, device_map="auto")

    else:
        model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_auth_token=model_args.use_auth_token,
    )


    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token


    #define hyper parameters
    training_args_sft = TrainingArguments(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size= training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=1.41e-5,
        logging_steps=1,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=-1,
        report_to=None,
        save_steps=5000,
        save_total_limit=3,
        push_to_hub=False,
        hub_model_id=None,
        bf16=True,
    )
  

    # Initialize our Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args_sft,
        max_seq_length=512,
        tokenizer=tokenizer,
        train_dataset=train_eval_dataset['train'],
        eval_dataset=train_eval_dataset['eval'],
        dataset_text_field=model_args.dataset_text_field,
    )

    # Training
    trainer.train()
    trainer.save_model(training_args.output_dir)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
