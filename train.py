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
from transformers.integrations import WandbCallback

from simcse.data_configs import load_data_config
from simcse.models_v2 import PretrainedModelForContrastiveLearning

from simcse.trainers import CLTrainer
from simcse.utils import wandb_setup
from simcse.data_process import passage_prepare_features, document_prepare_features, PassageDataCollatorWithPadding, \
    hfdataset_prepare_features
from src.arguments import ModelArguments, CustomTrainingArguments, ExtHFTrainingArguments, MoCoArguments
from src.moco import MoCo

logger = logging.getLogger(__name__)


@record
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, CustomTrainingArguments, ExtHFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args, hftraining_args, remaining_strings = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), return_remaining_strings=True)
    else:
        model_args, training_args, hftraining_args, remaining_strings = parser.parse_args_into_dataclasses(return_remaining_strings=True)

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
            model_args, _, _, moco_args = torch.load(os.path.join(hftraining_args.output_dir, "model_data_training_args.bin"))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(hftraining_args.local_rank) else logging.WARN,
    )

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
    logger.info("Training/evaluation parameters %s", hftraining_args)

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
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "hidden_dropout_prob": model_args.hidden_dropout_prob,
        "attention_probs_dropout_prob": model_args.attention_probs_dropout_prob,
    }

    if model_args.config_name:
        hfconfig = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        hfconfig = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        hfconfig = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if hftraining_args.do_train and training_args.train_file:
        data_files = {}
        data_files["train"] = training_args.train_file
        if data_files["train"] == 'c4':
            # https://huggingface.co/datasets/c4
            loaded_datasets = datasets.load_dataset("c4", "en", cache_dir=model_args.cache_dir)
            title_field = None
            text_field = 'text'
            train_dataset = loaded_datasets['train']
        elif data_files["train"] == 'cc100':
            # https://huggingface.co/datasets/cc100, examples are sentences, no longer than 100 words.
            loaded_datasets = datasets.load_dataset("cc100", lang="en", cache_dir=model_args.cache_dir, split='train[:2%]')
            title_field = None
            text_field = 'text'
            train_dataset = loaded_datasets
        elif data_files["train"] == 'the_pile':
            # https://huggingface.co/datasets/the_pile
            loaded_datasets = datasets.load_dataset("the_pile", cache_dir=model_args.cache_dir, split='train[:1%]')
            title_field = None
            text_field = 'text'
            train_dataset = loaded_datasets
        elif data_files["train"] == 'wikipedia':
            # https://huggingface.co/datasets/wikipedia
            loaded_datasets = datasets.load_dataset("wikipedia", "20220301.en", cache_dir=model_args.cache_dir)
            title_field = 'title'
            text_field = 'text'
            train_dataset = loaded_datasets['train']
        else:
            loaded_datasets = datasets.load_dataset("text",
                                                    data_files=data_files,
                                                    keep_in_memory=False,
                                                    cache_dir=model_args.cache_dir)
            train_dataset = loaded_datasets['train']

        if training_args.data_type == 'document':
            data_prep_config = load_data_config(training_args)
            parse_fn = partial(document_prepare_features,
                               tokenizer=tokenizer,
                               max_seq_length=training_args.max_seq_length,
                               padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
                               **data_prep_config
                               )
        elif training_args.data_type == 'hf':
            data_prep_config = load_data_config(training_args)
            parse_fn = partial(hfdataset_prepare_features, tokenizer=tokenizer,
                               max_seq_length=training_args.max_seq_length,
                               padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
                               title_field=title_field,
                               text_field=text_field,
                               **data_prep_config
            )
        train_dataset = train_dataset.shuffle(seed=hftraining_args.seed)
        train_dataset.set_transform(parse_fn)

        psg_parse_fn = partial(passage_prepare_features, tokenizer=tokenizer,
                               max_seq_length=training_args.max_seq_length,
                               padding_strategy='max_length' if training_args.pad_to_max_length else 'longest')

        # load a subset of wikipedia as devset
        if training_args.dev_file:
            dev_dataset = datasets.load_dataset("csv",
                                                data_files={"dev": training_args.dev_file},
                                                keep_in_memory=False,
                                                cache_dir=model_args.cache_dir,
                                                delimiter="\t" if "tsv" in training_args.dev_file else ",",
                                                split='dev')
            dev_dataset.set_transform(psg_parse_fn)
        else:
            dev_dataset = None
    else:
        train_dataset = None
        dev_dataset = None

    if model_args.arch_type == 'simcl':
        model = PretrainedModelForContrastiveLearning(
            hfconfig=hfconfig,
            model_args=model_args,
            seed=hftraining_args.seed
        )
    elif model_args.arch_type == 'moco':
        model = MoCo(moco_args, model_args, hfconfig)
    else:
        raise NotImplementedError('Unknown architecture name', model_args.arch_type)
    """""""""""""""""""""
    Set up trainer
    """""""""""""""""""""
    trainer = CLTrainer(
        model=model,
        args=hftraining_args,
        train_dataset=train_dataset if hftraining_args.do_train else None,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=PassageDataCollatorWithPadding(
            tokenizer,
            batch_size=hftraining_args.train_batch_size,
            max_length=training_args.max_seq_length,
            padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
            do_mlm=model_args.do_mlm,
            mlm_probability=training_args.mlm_probability)
    )
    trainer.model_args = model_args
    setattr(trainer, 'model_data_training_args', [model_args, training_args, hftraining_args, moco_args])
    torch.save(trainer.model_data_training_args, os.path.join(hftraining_args.output_dir, "model_data_training_args.bin"))

    # if it's a path, reload status
    if os.path.isdir(model_args.model_name_or_path):
        if model_args.arch_type == 'moco':
            state_dict = torch.load(os.path.join(model_args.model_name_or_path, 'pytorch_model.bin'), map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        trainer.state.load_from_json(os.path.join(model_args.model_name_or_path, "trainer_state.json"))
        wandb_resume = True
    else:
        wandb_resume = False

    # override wandb setup to log customized hyperparameters
    wandb_callback = [ch for ch in trainer.callback_handler.callbacks if isinstance(ch, WandbCallback)]
    if wandb_callback:
        # override wandb_callback's setup method to record our customized hyperparameters
        wandb_callback = wandb_callback[0]
        new_setup = functools.partial(wandb_setup, model_args=model_args,
                                      data_args=training_args, moco_args=moco_args, resume=wandb_resume)
        wandb_callback.setup =types.MethodType(new_setup, wandb_callback)

    """""""""""""""""""""
    Start Training
    """""""""""""""""""""
    if hftraining_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
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
        if training_args.beir_datasets:
            final_beir_datasets = training_args.beir_datasets
        else:
            final_beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020',
                                   'cqadupstack', 'trec-covid',
                                   'quora', 'nq', 'dbpedia-entity', 'hotpotqa']
        results = trainer.evaluate_beir(epoch=trainer.state.epoch,
                                        output_dir=hftraining_args.output_dir,
                                        sim_function=model_args.sim_type,
                                        beir_datasets=final_beir_datasets)
        results_senteval = trainer.evaluate_senteval(epoch=trainer.state.epoch, output_dir=hftraining_args.output_dir)
        results.update(results_senteval)

        output_eval_file = os.path.join(hftraining_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
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
