import functools
import logging
import os
import sys
from enum import Enum

import torch
import datasets
import types

from functools import partial
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Dict, Tuple
import torch

import datasets

import transformers
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BertForPreTraining,
)
from transformers.trainer_utils import is_main_process
from transformers.integrations import WandbCallback
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from simcse.models import BertForCL, PretrainedModelForContrastiveLearningV1
from simcse.trainers import CLTrainer
from simcse.utils import wandb_setup
from simcse.data_process import passage_prepare_features, document_prepare_features, PassageDataCollatorWithPadding

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # @memray
    cl_loss_weights: str = field(
        default=None,
        metadata={
            "help": "Whether the parameters of query/doc encoder are shared."
                    ""
        },
    )
    shared_encoder: bool = field(
        default=False,
        metadata={
            "help": "Whether the parameters of query/doc encoder are shared."
        },
    )
    hidden_dropout_prob: float = field(
        default=0.10,
        metadata={
            "help": "."
        }
    )
    attention_probs_dropout_prob: float = field(
        default=0.10,
        metadata={
            "help": "."
        }
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    )
    q_proj_type: str = field(
        default="",
        metadata={
            "help": "projector MLP setting, format is"
                    "``: no projecter"
                    "mlp: a simple D by D dense layer with Tanh activation, no parameter sharing (used in SimCSE)"
                    "1024-2048: three dense layers (D*1024*2048) with BatchNorm1d and ReLU (barlow-twin)"
        }
    )
    d_proj_type: str = field(
        default="",
        metadata={
            "help": "projector MLP setting, format is"
                    "shared: use the same parameters as q_proj"
                    "``: no projecter"
                    "mlp: a simple D by D dense layer with Tanh activation, no parameter sharing (used in SimCSE)"
                    "1024-2048: three dense layers (D*1024*2048) with BatchNorm1d and ReLU (barlow-twin)"
        }
    )
    sim_type: str = field(
        default="dot",
        metadata={
            "help": "What similarity metric function to use (dot, cosine)."
        }
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # @memray
    # Training
    data_type: str = field(
        default=None,
        metadata={
            "help": "document or passage."
        }
    )
    data_pipeline_name: str = field(
        default=None,
        metadata={
            "help": "Pre-defined data pipeline name. If set, all data hyper-parameters below will be overwritten."
        }
    )
    max_psg_len: int = field(
        default=None,
        metadata={
            "help": "if data_type is document and max_psg_len is given, we first randomly crop a contiguous span, "
                    "and Q/D will be sampled from it."
        },
    )
    min_dq_len: int = field(
        default=None,
        metadata={
            "help": "The minimal number of words for sampled query and doc."
        },
    )
    min_q_len: float = field(
        default=None,
        metadata={
            "help": "min Query len. If less 1.0, it denotes a length ratio."
        },
    )
    max_q_len: float = field(
        default=None,
        metadata={
            "help": "max Query len. If less 1.0, it denotes a length ratio."
        },
    )
    min_d_len: float = field(
        default=None,
        metadata={
            "help": "min Doc len. If less 1.0, it denotes a length ratio."
        },
    )
    max_d_len: float = field(
        default=None,
        metadata={
            "help": "max Doc len. If less 1.0, it denotes a length ratio."
        },
    )
    word_del_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Ratio for applying word deletion, for both Q and D."
        },
    )
    query_in_doc: bool = field(
        default=False,
        metadata={
            "help": "Whether sampled query must appear in doc. "
        },
    )
    q_retain_ratio: float = field(
        default=0.0,
        metadata={
            "help": "For ICT Q is taken and removed from D, this ratio controls by which rate Q is retained in D."
        },
    )
    section_or_paragraph: str = field(
        default='paragraph',
        metadata={
            "help": "Sample section or paragraph while loading wiki data."
        },
    )
    include_title_ratio: float = field(
        default=0.0,
        metadata={
            "help": "whether doc title is added (at the beginning)."
        },
    )
    # BEIR test
    beir_datasets: List[str] = field(
        default=None,
        metadata={
            "help": "Specify what BEIR datasets will be used in evaluation."
                    "Only affect the do_test phrase, not effective for during-training evaluation."
        }
    )
    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            print('Neither dataset_name or train_file is set, which is only allowed for testing.')
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "tsv", "jsonl", "json", "txt"], "`train_file` should be a csv, a json or a txt file."

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ExtHFTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

data_pipelines = {
    'psg-32-identical': {
        'max_psg_len': 32,
        'min_dq_len': 8,
        'min_q_len': 1,
        'max_q_len': 1,
        'min_d_len': 1,
        'max_d_len': 1,
        'word_del_ratio': 0,
        'query_in_doc': True,
        'q_retain_ratio': 1,
        'section_or_paragraph': 'paragraph',
        'include_title_ratio': 0.0,
    },
    'psg-128-notitle': {
        'max_psg_len': 128,
        'min_dq_len': 10,
        'min_q_len': 8,
        'max_q_len': 64,
        'min_d_len': 1,
        'max_d_len': 1,
        'word_del_ratio': 0,
        'query_in_doc': True,
        'q_retain_ratio': 1,
        'section_or_paragraph': 'paragraph',
        'include_title_ratio': 0.0,
    },
    'psg-64-notitle': {
        'max_psg_len': 64,
        'min_dq_len': 10,
        'min_q_len': 10,
        'max_q_len': 1,
        'min_d_len': 1,
        'max_d_len': 1,
        'word_del_ratio': 0,
        'query_in_doc': True,
        'q_retain_ratio': 1,
        'section_or_paragraph': 'paragraph',
        'include_title_ratio': 0.0,
    },
    'psg-32-notitle': {
        'max_psg_len': 32,
        'min_dq_len': 10,
        'min_q_len': 10,
        'max_q_len': 1,
        'min_d_len': 1,
        'max_d_len': 1,
        'word_del_ratio': 0,
        'query_in_doc': True,
        'q_retain_ratio': 1,
        'section_or_paragraph': 'paragraph',
        'include_title_ratio': 0.0,
    },
    'ICT-128-notitle': {
        'max_psg_len': 128,
        'min_dq_len': 10,
        'min_q_len': 0.05,
        'max_q_len': 0.25,
        'min_d_len': 1,
        'max_d_len': 1,
        'word_del_ratio': 0,
        'query_in_doc': True,
        'q_retain_ratio': 0.1,
        'section_or_paragraph': 'paragraph',
        'include_title_ratio': 0.0,
    },
    'ICT-256-notitle': {
        'max_psg_len': 256,
        'min_dq_len': 10,
        'min_q_len': 0.05,
        'max_q_len': 0.25,
        'min_d_len': 1,
        'max_d_len': 1,
        'word_del_ratio': 0,
        'query_in_doc': True,
        'q_retain_ratio': 0.1,
        'section_or_paragraph': 'paragraph',
        'include_title_ratio': 0.0,
    },
    'ConTr-128-notitle': {
        'max_psg_len': 128,
        'min_dq_len': 10,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'word_del_ratio': 0.1,
        'query_in_doc': False,
        'q_retain_ratio': 0.0,
        'section_or_paragraph': 'paragraph',
        'include_title_ratio': 0.0,
    },
    'ConTr-256-notitle': {
        'max_psg_len': 256,
        'min_dq_len': 10,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'word_del_ratio': 0.1,
        'query_in_doc': False,
        'q_retain_ratio': 0.0,
        'section_or_paragraph': 'paragraph',
        'include_title_ratio': 0.0,
    },
    'test-256': {
        'max_psg_len': 256,
        'min_dq_len': 10,
        'min_q_len': 1.0,
        'max_q_len': 1.0,
        'min_d_len': 1.0,
        'max_d_len': 1.0,
        'word_del_ratio': 0.0,
        'query_in_doc': True,
        'q_retain_ratio': 0.5,
        'section_or_paragraph': 'section',
        'include_title_ratio': 1.0,
    }
}

@record
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ExtHFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
        handlers=[
            logging.FileHandler(training_args.output_dir+"/train_log.txt", mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

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
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
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

    """""""""""""""""""""
    Load training dataset
    """""""""""""""""""""
    if False:
        model = BertForCL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args
        )
    else:
        if not os.path.exists(model_args.model_name_or_path):
            model = PretrainedModelForContrastiveLearningV1(
                config=config,
                model_args=model_args
            )
        else:
            arg_path = os.path.join(model_args.model_name_or_path, "model_data_training_args.bin")
            if os.path.exists(arg_path):
                _model_args, _, _ = torch.load(arg_path)
                _model_args.model_name_or_path = model_args.model_name_or_path
                model_args = _model_args
            else:
                # awkward backward compatibility
                if 'avg' in model_args.model_name_or_path:
                    model_args.pooler_type = 'avg'
                if 'cosine' in model_args.model_name_or_path:
                    model_args.sim_type = 'cosine'
            model = PretrainedModelForContrastiveLearningV1.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                model_args=model_args
            )
    if model_args.do_mlm:
        pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
        model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())

    """""""""""""""""""""
    Load training dataset
    """""""""""""""""""""
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if training_args.do_train:
        # load training data
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        if extension == "jsonl" or extension == "json":
            extension = "text"
            # extension = "json" # json.decoder.JSONDecodeError: Extra data: line 2 column 1 (char 44001)
        if extension == "csv" or extension == "tsv":
            loaded_datasets = datasets.load_dataset("csv",
                                    data_files=data_files,
                                    delimiter="\t" if "tsv" in data_args.train_file else ",",
                                    keep_in_memory=False,
                                    cache_dir=model_args.cache_dir,
                                    )
        else:
            loaded_datasets = datasets.load_dataset(extension,
                                                    data_files=data_files,
                                                    keep_in_memory=False,
                                                    cache_dir=model_args.cache_dir)

        # prepare for data loader
        if data_args.data_pipeline_name:
            data_prep_config = data_pipelines[data_args.data_pipeline_name]
            logger.info('Using pre-defined data pipeline: ' + str(data_args.data_pipeline_name))
        else:
            data_prep_config = {
                'max_psg_len': data_args.max_psg_len,
                'min_dq_len': data_args.min_dq_len,
                'min_q_len': data_args.min_q_len,
                'max_q_len': data_args.max_q_len,
                'min_d_len': data_args.min_d_len,
                'max_d_len': data_args.max_d_len,
                'word_del_ratio': data_args.word_del_ratio,
                'query_in_doc': data_args.query_in_doc,
                'q_retain_ratio': data_args.q_retain_ratio,
                'section_or_paragraph': data_args.section_or_paragraph,
                'include_title_ratio': data_args.include_title_ratio,
            }
        if data_args.data_type == 'document':
            logger.info('Data loading parameters:')
            for k, v in data_prep_config.items():
                setattr(data_args, k, v)
                logger.info(f'\t\t{k} = {v}')
            parse_fn = partial(document_prepare_features,
                               tokenizer=tokenizer,
                               max_seq_length=data_args.max_seq_length,
                               padding_strategy='max_length' if data_args.pad_to_max_length else 'longest',
                               **data_prep_config
                               )
        elif data_args.data_type == 'passage':
            parse_fn = partial(passage_prepare_features, tokenizer=tokenizer,
                               max_seq_length=data_args.max_seq_length,
                               padding_strategy='max_length' if data_args.pad_to_max_length else 'longest')
        else:
            f'Unknown data_args.data_type=[{data_args.data_type}]'
        train_dataset = loaded_datasets['train']
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        train_dataset.set_transform(parse_fn)

    """""""""""""""""""""
    Set up trainer
    """""""""""""""""""""
    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=PassageDataCollatorWithPadding(
            tokenizer,
            max_length=data_args.max_seq_length,
            padding_strategy='max_length' if data_args.pad_to_max_length else 'longest',
            do_mlm=model_args.do_mlm,
            mlm_probability=data_args.mlm_probability)
    )
    trainer.model_args = model_args
    setattr(trainer, 'model_data_training_args', [model_args, data_args, training_args])

    """""""""""""""""""""
    Start Training
    """""""""""""""""""""
    if training_args.do_train:
        wandb_callback = [ch for ch in trainer.callback_handler.callbacks if isinstance(ch, WandbCallback)]
        if wandb_callback:
            # override wandb_callback's setup method to record our customized hyperparameters
            wandb_callback = wandb_callback[0]
            new_setup = functools.partial(wandb_setup, model_args=model_args, data_args=data_args)
            wandb_callback.setup =types.MethodType(new_setup, wandb_callback)
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    """""""""""""""""""""
    Start Evaluation
    """""""""""""""""""""
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate_beir(epoch=trainer.state.epoch,
                                        output_dir=training_args.output_dir,
                                        sim_function=model_args.sim_type,
                                        beir_datasets=data_args.beir_datasets)
        results_senteval = trainer.evaluate_senteval(epoch=trainer.state.epoch, output_dir=training_args.output_dir)
        results.update(results_senteval)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
