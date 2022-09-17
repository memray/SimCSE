import functools
import logging
import os
import sys
import types

from functools import partial

import datasets
import torch

import transformers
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.integrations import WandbCallback, rewrite_logs

from simcse.data_loader import load_datasets

from simcse.trainers import CLTrainer
from simcse.utils import wandb_setup, wandb_setup_eval
from simcse.data_process import PassageDataCollatorWithPadding
from src.arguments import CustomTrainingArguments, ExtHFTrainingArguments, MoCoArguments
# from src.finetune.finetuning import load_finetuning_datasets
from src.moco import MoCo

logger = logging.getLogger(__name__)


@record
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((CustomTrainingArguments, ExtHFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args, hftraining_args, remaining_strings = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), return_remaining_strings=True)
    else:
        training_args, hftraining_args, remaining_strings = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    moco_args = MoCoArguments().parse()

    # Setup logging
    os.makedirs(hftraining_args.output_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(hftraining_args.local_rank) else logging.WARN,
        handlers=[
            logging.FileHandler(hftraining_args.output_dir+"/train_log.txt", mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
        logger.warning(f"Input arguments: \n\t {' '.join(sys.argv)}")

    if (
        os.path.exists(hftraining_args.output_dir)
        and os.listdir(hftraining_args.output_dir)
    ):
        if hftraining_args.do_train and not hftraining_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({hftraining_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        elif training_args.resume_training and os.path.exists(os.path.join(hftraining_args.output_dir, "model_data_training_args.bin")):
            logger.info("Reloading moco_args from %s",
                        os.path.join(hftraining_args.output_dir, "model_data_training_args.bin"))
            _, _, moco_args = torch.load(os.path.join(hftraining_args.output_dir, "model_data_training_args.bin"))

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {hftraining_args.local_rank}, device: {hftraining_args.device}, n_gpu: {hftraining_args.n_gpu}"
        + f" distributed training: {bool(hftraining_args.local_rank != -1)}, 16-bits training: {hftraining_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(hftraining_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
        logger.info("*" * 50)
        logger.info("Training/evaluation parameters:\n%s", hftraining_args)
        logger.info("Custom training parameters:\n%s", training_args)
        logger.info("MoCo model parameters:\n%s", moco_args)
        logger.info("*" * 50)

    # Set seed before initializing model.
    set_seed(hftraining_args.seed)

    """""""""""""""""""""
    Load HF configs and tokenizer
    """""""""""""""""""""
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": training_args.cache_dir,
        "hidden_dropout_prob": moco_args.hidden_dropout_prob,
        "attention_probs_dropout_prob": moco_args.attention_probs_dropout_prob,
    }
    hf_config = AutoConfig.from_pretrained(moco_args.model_name_or_path, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(moco_args.model_name_or_path, cache_dir=training_args.cache_dir, use_fast=True)
    model = MoCo(moco_args, hf_config)
    if training_args.fine_tuning:
        train_dataset, train_collator = None, None
        dev_dataset = None
    else:
        train_dataset, dev_dataset = load_datasets(tokenizer, training_args, hftraining_args, moco_args)
        train_collator = PassageDataCollatorWithPadding(
            tokenizer,
            batch_size=hftraining_args.train_batch_size,
            padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
            max_length=training_args.max_seq_length,
            cap_qd_tokens=training_args.cap_qd_tokens,
            max_q_tokens=training_args.max_q_tokens
        )
    """""""""""""""""""""
    Set up trainer
    """""""""""""""""""""
    trainer = CLTrainer(
        model=model,
        args=hftraining_args,
        train_dataset=train_dataset if hftraining_args.do_train else None,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=train_collator
    )
    trainer.training_args = training_args
    trainer.hftraining_args = hftraining_args
    trainer.moco_args = moco_args
    trainer.best_score = -10000.0
    early_stop_patience = sys.maxsize
    trainer.early_stop_patience = early_stop_patience
    trainer.early_stop_counter = 0
    setattr(trainer, 'model_data_training_args', [training_args, hftraining_args, moco_args])
    torch.save(trainer.model_data_training_args, os.path.join(hftraining_args.output_dir, "model_data_training_args.bin"))
    # if it's a path, reload status
    if os.path.isdir(moco_args.model_name_or_path):
        state_dict = torch.load(os.path.join(moco_args.model_name_or_path, 'pytorch_model.bin'), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        trainer.state = trainer.state.load_from_json(os.path.join(moco_args.model_name_or_path, "trainer_state.json"))
    # override wandb setup to log customized hyperparameters
    wandb_callbacks = [ch for ch in trainer.callback_handler.callbacks if isinstance(ch, WandbCallback)]
    wandb_callback = wandb_callbacks[0] if wandb_callbacks else None
    """""""""""""""""""""
    Start Training
    """""""""""""""""""""
    if hftraining_args.do_train:
        if trainer.is_world_process_zero() and wandb_callback:
            # override wandb_callback's setup method to record our customized hyperparameters
            new_setup = functools.partial(wandb_setup,
                                          hftraining_args=hftraining_args,
                                          training_args=training_args,
                                          moco_args=moco_args, resume=False)
            wandb_callback.setup = types.MethodType(new_setup, wandb_callback)

        model_path = (
            moco_args.model_name_or_path
            if (moco_args.model_name_or_path is not None and os.path.isdir(moco_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(hftraining_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(hftraining_args.output_dir, "trainer_state.json"))

    """""""""""""""""""""
    Start Evaluation
    """""""""""""""""""""
    results = {}
    if hftraining_args.do_eval:
        logger.info("*** Evaluate ***")
        if trainer.is_world_process_zero() and wandb_callback and \
                (not wandb_callback._initialized or wandb_callback._wandb.run is None):
            wandb_setup_eval(wandb_callback, hftraining_args)
        if training_args.beir_datasets:
            final_beir_datasets = training_args.beir_datasets
        else:
            final_beir_datasets = ['msmarco', 'trec-covid', 'nfcorpus', 'nq', 'hotpotqa', 'fiqa', 'arguana', 'webis-touche2020', 'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact', 'quora', 'cqadupstack']
            # 11 tests, ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'trec-covid', 'quora', 'nq', 'dbpedia-entity']

        results = trainer.evaluate_beir(epoch=trainer.state.epoch,
                                        output_dir=hftraining_args.output_dir,
                                        batch_size=training_args.beir_batch_size,
                                        sim_function=moco_args.sim_metric,
                                        beir_datasets=final_beir_datasets)
        results_senteval = trainer.evaluate_senteval(epoch=trainer.state.epoch, output_dir=hftraining_args.output_dir)
        results.update(results_senteval)
        if trainer.is_world_process_zero():
            if wandb_callback and wandb_callback._wandb.run:
                results = rewrite_logs(results)
                if trainer.state.global_step:
                    results['train/global_step'] = trainer.state.global_step
                wandb_callback._wandb.log({**results})
            output_eval_file = os.path.join(hftraining_args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(hftraining_args.output_dir, "trainer_state.json"))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
