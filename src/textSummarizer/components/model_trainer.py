import os
import torch

from dataclasses import dataclass
from pathlib import Path

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

from datasets import load_from_disk

from textSummarizer.constants import *
from textSummarizer.utils.common import (
    read_yaml,
    create_directories
)
from textSummarizer.entity import ModelTrainerConfig


class ModelTrainer:

    def __init__(
        self,
        config: ModelTrainerConfig
    ):

        self.config = config


    def train(self):

        # FORCE CPU FOR STABILITY
        device = "cpu"

        print(f"Using device: {device}")


        # LOAD TOKENIZER
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_ckpt
        )


        # LOAD MODEL
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)



        # DATA COLLATOR
        seq2seq_data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model_pegasus
        )


        # LOAD DATASET
        dataset_samsum_pt = load_from_disk(
            self.config.data_path
        )


        # SMALL DATASET FOR DEBUGGING
        train_data = dataset_samsum_pt["train"].select(
            range(2)
        )

        eval_data = dataset_samsum_pt["validation"].select(
            range(2)
        )


        # TRAINING ARGUMENTS
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,

            num_train_epochs=1,

            warmup_steps=10,

            per_device_train_batch_size=1,

            weight_decay=0.01,

            logging_steps=1,

            save_steps=500,

            gradient_accumulation_steps=1,

            report_to="none"
        )


        # TRAINER
        trainer = Trainer(
            model=model_pegasus,

            args=trainer_args,

            data_collator=seq2seq_data_collator,

            train_dataset=train_data,

            eval_dataset=eval_data
        )


        # START TRAINING
        trainer.train()


        # SAVE MODEL
        model_pegasus.save_pretrained(
            os.path.join(
                self.config.root_dir,
                "pegasus-samsum-model"
            )
        )


        # SAVE TOKENIZER
        tokenizer.save_pretrained(
            os.path.join(
                self.config.root_dir,
                "tokenizer"
            )
        )


        print("\nTraining completed successfully.")