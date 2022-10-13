import functools
import logging
import os
import shutil
import sys
import types
import warnings

import torch

import transformers
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.integrations import WandbCallback, rewrite_logs

from src import eval_utils
from src.data_loader import load_datasets
from src.finetune_data import load_finetune_dataset
from src.inbatch import InBatch

from src.trainer import DenseRetrievalTrainer

from src.training_utils import wandb_setup, wandb_setup_eval, reload_model_from_ckpt, reload_model_from_pretrained
from src.data_process import PassageDataCollatorWithPadding
from src.arguments import CustomTrainingArguments, HFTrainingArguments, MoCoArguments
from src.moco import MoCo

logger = logging.getLogger(__name__)


@record
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((CustomTrainingArguments, HFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args, hftraining_args, remaining_strings = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), return_remaining_strings=True)
    else:
        training_args, hftraining_args, remaining_strings = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    warnings.warn('remaining_strings:' + str(remaining_strings))

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
        elif os.path.exists(os.path.join(hftraining_args.output_dir, "model_data_training_args.bin")):
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
    if moco_args.arch_type == 'moco':
        model = MoCo(moco_args, hf_config)
    elif moco_args.arch_type == 'inbatch':
        model = InBatch(moco_args, hf_config)
    else:
        raise NotImplementedError(f'Unknown arch type {hf_config.arch_type}')
    if training_args.finetune:
        train_dataset, dev_dataset = load_finetune_dataset(tokenizer, training_args, hftraining_args, moco_args)
        train_collator = PassageDataCollatorWithPadding(
            tokenizer,
            batch_size=hftraining_args.train_batch_size,
            padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
            max_length=training_args.max_seq_length,
            max_q_tokens=training_args.max_q_tokens,
            max_d_tokens=training_args.max_d_tokens
        )
    else:
        train_dataset, dev_dataset = load_datasets(tokenizer, training_args, hftraining_args, moco_args)
        train_collator = PassageDataCollatorWithPadding(
            tokenizer,
            batch_size=hftraining_args.train_batch_size,
            padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
            max_length=training_args.max_seq_length,
            max_q_tokens=training_args.max_q_tokens,
            max_d_tokens=training_args.max_d_tokens
        )

    """""""""""""""""""""
    Set up trainer
    """""""""""""""""""""
    trainer = DenseRetrievalTrainer(
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
    if training_args.reload_model_from:
        if os.path.isdir(training_args.reload_model_from):
            reload_model_from_ckpt(model, training_args.reload_model_from)
            # trainer.state = trainer.state.load_from_json(os.path.join(training_args.reload_model_from, "trainer_state.json"))
        elif training_args.reload_model_from.startswith('facebook/'):
            reload_model_from_pretrained(model, training_args.reload_model_from)
        else:
            raise Exception('R U sure?')
    # override wandb setup to log customized hyperparameters
    wandb_callbacks = [ch for ch in trainer.callback_handler.callbacks if isinstance(ch, WandbCallback)]
    wandb_callback = wandb_callbacks[0] if wandb_callbacks else None
    """""""""""""""""""""
    Start Training
    """""""""""""""""""""
    if hftraining_args.do_train:
        if trainer.is_world_process_zero() and wandb_callback:
            # remove previous wandb outputs
            if os.path.exists(hftraining_args.output_dir+'/wandb'):
                shutil.rmtree(hftraining_args.output_dir+'/wandb')
                os.makedirs(hftraining_args.output_dir+'/wandb')
            # override wandb_callback's setup method to record our customized hyperparameters
            new_setup = functools.partial(wandb_setup,
                                          hftraining_args=hftraining_args,
                                          training_args=training_args,
                                          moco_args=moco_args, resume=False)
            wandb_callback.setup = types.MethodType(new_setup, wandb_callback)

        train_result = trainer.train()
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
        # eval BEIR
        final_beir_datasets = ['msmarco', 'dbpedia-entity', 'fever', 'climate-fever', 'nq', 'hotpotqa',
                               'quora', 'cqadupstack', 'trec-covid', 'arguana', 'webis-touche2020',
                               'scidocs', 'scifact', 'nfcorpus', 'fiqa']
        if trainer.is_world_process_zero():
            try:
                prev_eval_dir = os.path.join(hftraining_args.output_dir, 'eval_output', 'checkpoint-%d' % (hftraining_args.max_steps))
                logger.info("Attempt to copy previous beir output: %s" % (prev_eval_dir))
                shutil.copytree(prev_eval_dir, hftraining_args.output_dir+'/beir_output', dirs_exist_ok=True)
            except Exception as e:
                logger.info("Error, failed to copy previous beir output: %s (exist? %s) : %s" % (prev_eval_dir, os.path.exists(prev_eval_dir), e.strerror))
        try:
            prev_eval_dir = os.path.join(hftraining_args.output_dir, 'eval_output', 'checkpoint-%d' % (hftraining_args.max_steps))
            prev_dones = [f[:-5] for f in os.listdir(prev_eval_dir) if f.endswith('.json')]
            final_beir_datasets = [dataset for dataset in final_beir_datasets if dataset not in prev_dones]
            logger.info("Found previous beir output, new BEIR datasets: %s" % (str(final_beir_datasets)))
        except Exception as e:
            logger.info("Error, didn't find previous beir output: %s (exist? %s) : %s" % (prev_eval_dir, os.path.exists(prev_eval_dir), e.strerror))
            pass
        results_beir = eval_utils.evaluate_beir(
            model, tokenizer,
            beir_path=training_args.beir_path,
            sim_function=model.sim_metric,
            add_qd_prompt=(training_args.dq_prompt_ratio > 0.0),
            batch_size=training_args.beir_batch_size,
            beir_datasets=final_beir_datasets,
            output_dir=hftraining_args.output_dir+'/beir_output',
        )
        results.update(results_beir)
        # eval SentEval
        results_senteval = eval_utils.evaluate_senteval(model, tokenizer,
                                                        eval_senteval_transfer=True,
                                                        output_dir=hftraining_args.output_dir+'/senteval_output',
                                                        )
        results.update(results_senteval)
        # eval QA
        passages = eval_utils.load_passages(training_args.wiki_passage_path)
        embed_dir = os.path.join(hftraining_args.output_dir, 'wiki_emb')
        eval_utils.generate_passage_embeddings(model, tokenizer, passages, embed_dir)
        if trainer.is_world_process_zero():
            results_qa = eval_utils.evaluate_qa(model, tokenizer,
                                                passages=passages,
                                                qa_datasets_path=training_args.qa_datasets_path,
                                                passages_embeddings_path=embed_dir + '/*',
                                                encode_batch_size=training_args.qa_batch_size,
                                                output_dir=hftraining_args.output_dir+'/qa_output'
                                                )
            results.update(results_qa)
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
