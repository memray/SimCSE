from pathlib import Path

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed, IntervalStrategy,
)
import argparse
import os

from transformers.debug_utils import DebugOption
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available, \
    is_torch_bf16_available, is_torch_tf32_available, get_full_repo_name
from transformers.trainer_utils import EvaluationStrategy, HubStrategy, ShardedDDPOption
from transformers.training_args import default_logdir, trainer_log_levels, OptimizerNames

from simcse.trainers import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Dict, Tuple
import torch

from transformers.utils import logging

logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."})
    model_type: Optional[str] = field(default=None, metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)})
    arch_type: Optional[str] = field(default=None, metadata={"help": "moco or simcl"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."})
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})
    use_auth_token: bool = field(default=False, metadata={"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."})
    # @memray
    cl_loss_weights: str = field(default=None, metadata={"help": "Whether the parameters of query/doc encoder are shared."})
    shared_encoder: bool = field(default=False, metadata={"help": "Whether the parameters of query/doc encoder are shared."})
    hidden_dropout_prob: float = field(default=0.10, metadata={"help": "."})
    attention_probs_dropout_prob: float = field(default=0.10, metadata={"help": "."})

    # MoCo related arguments
    memory_type: str = field(default=None, metadata={"help": ""})
    memory_size: int = field(default=0, metadata={"help": ""})

    # SimCSE's arguments
    temp: float = field(default=0.05, metadata={"help": "Temperature for softmax."})
    # pooler_type: str = field(default="cls", metadata={"help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."})
    q_proj_type: str = field(default="none", metadata={"help": "projector MLP setting, format is"
                    "`none`: no projecter"
                    "mlp: a simple D by D dense layer with Tanh activation, no parameter sharing (used in SimCSE)"
                    "1024-2048: three dense layers (D*1024*2048) with BatchNorm1d and ReLU (barlow-twin)"})
    d_proj_type: str = field(default="none", metadata={"help": "projector MLP setting, format is"
                    "shared: use the same parameters as q_proj"
                    "`none`: no projecter"
                    "mlp: a simple D by D dense layer with Tanh activation, no parameter sharing (used in SimCSE)"
                    "1024-2048: three dense layers (D*1024*2048) with BatchNorm1d and ReLU (barlow-twin)"})
    sim_type: str = field(default="dot", metadata={"help": "What similarity metric function to use (dot, cosine)."})
    hard_negative_weight: float = field(default=0, metadata={"help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."})
    do_mlm: bool = field(default=False, metadata={"help": "Whether to use MLM auxiliary objective."})
    mlm_weight: float = field(default=0.1, metadata={"help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."})
    mlp_only_train: bool = field(default=False, metadata={"help": "Use MLP only during training"})

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
class CustomTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(default=5, metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"})
    # @memray
    # Training
    resume_training: str = field(default=None, metadata={"help": "resume training."})
    data_type: str = field(default=None, metadata={"help": "document or passage."})
    data_pipeline_name: str = field(default=None, metadata={"help": "Pre-defined data pipeline name. If set, all data hyper-parameters below will be overwritten."})
    max_context_len: int = field(default=None, metadata={"help": "if data_type is document and max_context_len is given, we first randomly crop a contiguous span, and Q/D will be sampled from it."})
    min_dq_len: int = field(default=None, metadata={"help": "The minimal number of words for sampled query and doc."})
    min_q_len: float = field(default=None, metadata={"help": "min Query len. If less 1.0, it denotes a length ratio."})
    max_q_len: float = field(default=None, metadata={"help": "max Query len. If less 1.0, it denotes a length ratio."})
    min_d_len: float = field(default=None, metadata={"help": "min Doc len. If less 1.0, it denotes a length ratio."})
    max_d_len: float = field(default=None, metadata={"help": "max Doc len. If less 1.0, it denotes a length ratio."})
    word_del_ratio: float = field(default=0.0, metadata={"help": "Ratio for applying word deletion, for both Q and D."})
    query_in_doc: bool = field(default=False, metadata={"help": "Whether sampled query must appear in doc. "})
    q_retain_ratio: float = field(default=0.0, metadata={"help": "For ICT Q is taken and removed from D, this ratio controls by which rate Q is retained in D."})
    context_range: str = field(default='paragraph', metadata={"help": "Sample a chunk from a document/section/paragraph while loading wiki data."})
    dq_prompt_ratio: float = field(default=0.0, metadata={"help": "Randomly add a prefix to indicate the input is Q/D."})
    title_as_query_ratio: float = field(default=0.0, metadata={"help": "randomly use title as query."})
    include_doctitle_ratio: float = field(default=0.0, metadata={"help": "whether doc title is added (at the beginning)."})
    # BEIR test
    beir_path: Optional[str] = field(default="/export/home/data/beir", metadata={ "help": "Base directory of BEIR data."})
    beir_datasets: List[str] = field(default=None, metadata={"help": "Specify what BEIR datasets will be used in evaluation."
                    "Only affect the do_test phrase, not effective for during-training evaluation."})
    beir_batch_size: int = field(default=16, metadata={"help": "Specify batch size for BEIR datasets."})
    # SimCSE's arguments
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The training data file (.txt or .csv)."})
    train_prob: Optional[str] = field(default=None, metadata={"help": "The sampling probability for multiple datasets."})
    dev_file: Optional[str] = field(default=None, metadata={"help": "The dev data file (.txt or .csv)."})
    max_seq_length: Optional[int] = field(default=32, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."})
    mlm_probability: float = field(default=0.15, metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"})

    # def __post_init__(self):
    #     if self.dataset_name is None and self.train_file is None:
    #         print('Neither dataset_name or train_file is set, which is only allowed for testing.')
    #     else:
    #         if self.train_file is not None:
    #             extension = self.train_file.split(".")[-1]
    #             assert extension in ["csv", "tsv", "jsonl", "json", "txt"], "`train_file` should be a csv, a json or a txt file."

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
    eval_transfer: bool = field(default=False, metadata={"help": "Evaluate transfer task dev sets (in validation)."})
    lr_scheduler_type: SchedulerType = field(default="linear", metadata={"help": "The scheduler type to use."})
    num_cycles: float = field(default=0.5, metadata={"help": "."})
    lr_end: float = field(default=1e-7, metadata={"help": "."})
    power: float = field(default=1.0, metadata={"help": "."})

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda:
            device = torch.device("cpu")
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
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

    def __post_init__(self, warnings=None):
        # Handle --use_env option in torch.distributed.launch (local_rank not passed as an arg then).
        # This needs to happen before any call to self.device or self.n_gpu.
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank

        # convert to int
        self.log_level = trainer_log_levels[self.log_level]
        self.log_level_replica = trainer_log_levels[self.log_level_replica]

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        #  see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS \
                and (self.eval_steps is None or self.eval_steps == 0) and (self.eval_ratio == 0.0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires "
                    f"either non-zero --eval_steps or --eval_ratio or --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

        if self.fp16_backend and self.fp16_backend != "auto":
            warnings.warn(
                "`fp16_backend` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `half_precision_backend` instead",
                FutureWarning,
            )
            self.half_precision_backend = self.fp16_backend

        if (self.bf16 or self.bf16_full_eval) and not is_torch_bf16_available():
            raise ValueError("Your setup doesn't support bf16. You need Ampere GPU, torch>=1.10, cuda>=11.0")

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")
        if self.bf16:
            if self.half_precision_backend == "apex":
                raise ValueError(
                    " `--half_precision_backend apex`: bf16 is not supported by apex. Use `--half_precision_backend amp` instead"
                )
            if not (self.sharded_ddp == "" or not self.sharded_ddp):
                raise ValueError("sharded_ddp is not supported with bf16")

        self.optim = OptimizerNames(self.optim)
        if self.adafactor:
            warnings.warn(
                "`--adafactor` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--optim adafactor` instead",
                FutureWarning,
            )
            self.optim = OptimizerNames.ADAFACTOR

        if (
            is_torch_available()
            and (self.device.type != "cuda")
            and not (self.device.type == "xla" and "GPU_NUM_DEVICES" in os.environ)
            and (self.fp16 or self.fp16_full_eval or self.bf16 or self.bf16_full_eval)
        ):
            raise ValueError(
                "Mixed precision training with AMP or APEX (`--fp16` or `--bf16`) and half precision evaluation (`--fp16_full_eval` or `--bf16_full_eval`) can only be used on CUDA devices."
            )

        if is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                # no need to assert on else

        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from transformers.integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio during training"
            )

        if isinstance(self.sharded_ddp, bool):
            self.sharded_ddp = "simple" if self.sharded_ddp else ""
        if isinstance(self.sharded_ddp, str):
            self.sharded_ddp = [ShardedDDPOption(s) for s in self.sharded_ddp.split()]
        if self.sharded_ddp == [ShardedDDPOption.OFFLOAD]:
            raise ValueError(
                "`--sharded_ddp offload` can't work on its own. It needs to be added to `--sharded_ddp zero_dp_2` or "
                '`--sharded_ddp zero_dp_3`. For example, `--sharded_ddp "zero_dp_2 offload"`.'
            )
        elif len(self.sharded_ddp) > 1 and ShardedDDPOption.SIMPLE in self.sharded_ddp:
            raise ValueError("`--sharded_ddp simple` is not compatible with any other option.")
        elif ShardedDDPOption.ZERO_DP_2 in self.sharded_ddp and ShardedDDPOption.ZERO_DP_3 in self.sharded_ddp:
            raise ValueError("`--sharded_ddp zero_dp_2` is not compatible with `--sharded_ddp zero_dp_3`.")

        if self.tpu_metrics_debug:
            warnings.warn(
                "using `--tpu_metrics_debug` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--debug tpu_metrics_debug` instead",
                FutureWarning,
            )
            self.debug += " tpu_metrics_debug"
            self.tpu_metrics_debug = False
        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]

        if self.deepspeed:
            # - must be run very last in arg parsing, since it will use a lot of these settings.
            # - must be run before the model is created.
            from transformers.deepspeed import HfTrainerDeepSpeedConfig

            # will be used later by the Trainer
            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

        if self.push_to_hub_token is not None:
            warnings.warn(
                "`--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_token` instead.",
                FutureWarning,
            )
            self.hub_token = self.push_to_hub_token

        if self.push_to_hub_model_id is not None:
            self.hub_model_id = get_full_repo_name(
                self.push_to_hub_model_id, organization=self.push_to_hub_organization, token=self.hub_token
            )
            if self.push_to_hub_organization is not None:
                warnings.warn(
                    "`--push_to_hub_model_id` and `--push_to_hub_organization` are deprecated and will be removed in "
                    "version 5 of 🤗 Transformers. Use `--hub_model_id` instead and pass the full repo name to this "
                    f"argument (in this case {self.hub_model_id}).",
                    FutureWarning,
                )
            else:
                warnings.warn(
                    "`--push_to_hub_model_id` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                    "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                    f"{self.hub_model_id}).",
                    FutureWarning,
                )
        elif self.push_to_hub_organization is not None:
            self.hub_model_id = f"{self.push_to_hub_organization}/{Path(self.output_dir).name}"
            warnings.warn(
                "`--push_to_hub_organization` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                f"{self.hub_model_id}).",
                FutureWarning,
            )


class MoCoArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        # basic parameters
        self.parser.add_argument("--num_q_view", type=int, default=1)
        self.parser.add_argument("--num_k_view", type=int, default=1)
        self.parser.add_argument("--queue_strategy", type=str, default='fifo', help="'fifo', 'priority'")
        self.parser.add_argument("--num_extra_pos", type=int, default=0)
        self.parser.add_argument("--queue_size", type=int, default=65536)
        self.parser.add_argument('--pooling', type=str, default='average', help='average or cls')
        self.parser.add_argument("--pooling_dropout", type=str, default='none', help="none, standard, gaussian, variational")
        self.parser.add_argument("--pooling_dropout_prob", type=float, default=0.0, help="bernoulli, gaussian, variational")
        self.parser.add_argument('--merger_type', type=str, default=None, help="projector MLP setting, format is "
           "(1)`none`: no projecter; (2) multiview; (3)`mlp`: a simple D by D dense layer with Tanh activation, no parameter sharing (used in SimCSE) "
           "(4) `1024-2048`: multi-layer dense connections (D*1024*2048) with BatchNorm1d and ReLU (barlow-twin)")
        self.parser.add_argument("--warmup_queue_size_ratio", type=float, default=0.0, help='linearly increase queue size to 100% until training_steps*warmup_queuesize_ratio.')
        self.parser.add_argument("--queue_update_steps", type=int, default=1, help='we only update the model parameters (backprop) every k step, and but update queue in the rest k-1 steps.')
        self.parser.add_argument("--momentum", type=float, default=0.9995)
        self.parser.add_argument("--temperature", type=float, default=0.05)
        self.parser.add_argument('--label_smoothing', type=float, default=0.)
        self.parser.add_argument('--norm_query', action='store_true')
        self.parser.add_argument('--norm_doc', action='store_true')
        self.parser.add_argument('--moco_train_mode_encoder_k', action='store_true')
        # self.parser.add_argument("--score_function", type=str, default='dot', )
        self.parser.add_argument('--random_init', action='store_true', help='init model with random weights')
        self.parser.add_argument('--projection_size', type=int, default=768)
        self.parser.add_argument('--indep_encoder_k', type=bool, default=False, help='whether to use an independent/asynchronous encoder.')
        # alignment+uniformity
        self.parser.add_argument('--align_unif_loss', type=bool, default=False)
        self.parser.add_argument('--align_weight', type=float, default=0.0)
        self.parser.add_argument('--align_alpha', type=float, default=2)
        self.parser.add_argument('--unif_weight', type=float, default=0.0)
        self.parser.add_argument('--unif_t', type=float, default=2)

    def print_options(self, opt):
        message = ''
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: %s]' % str(default)
            message += f'{str(k):>40}: {str(v):<40}{comment}\n'
        print(message, flush=True)
        model_dir = os.path.join(opt.output_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(opt.output_dir, 'models'))
        file_name = os.path.join(opt.output_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        # opt = self.parser.parse_args()
        return opt
