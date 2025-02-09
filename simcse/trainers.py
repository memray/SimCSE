import collections
import math
import os
import json
import time
import warnings

from numpy import float64
from packaging import version
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import SequentialSampler
from transformers import Trainer
from transformers.integrations import hp_params
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION, get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup, get_constant_schedule, get_constant_schedule_with_warmup

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    TrainOutput,
    set_seed,
    speed_metrics, EvalLoopOutput,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_torch_tpu_available, ExplicitEnum,
)
from transformers.trainer_callback import (
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)
from tqdm.auto import tqdm

from transformers.utils import logging
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

from SentEval import senteval
from beir_eval import beir_utils, dist_utils
from src import utils
from src.utils import get_cosine_with_hard_decayed_restarts_schedule_with_warmup

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

def _model_unwrap(model: nn.Module) -> nn.Module:
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return _model_unwrap(model.module)
    else:
        return model

# Set path to SentEval
PATH_TO_DATA = '/export/share/ruimeng/project/search/simcse/SentEval/data/'

class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    COSINE_WITH_DECAYED_RESTARTS = "cosine_with_decayed_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"

TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.COSINE_WITH_DECAYED_RESTARTS: get_cosine_with_hard_decayed_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
}
import numpy as np

logger = logging.get_logger(__name__)

class CLTrainer(Trainer):
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = None,
        report_align_unif: bool = True,
        report_metrics: bool = True
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        """
        args = self.args
        # this requires manual set since train/eval use the same collate_fn
        dataloader.collate_fn.batch_size = self.args.per_device_eval_batch_size
        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        model.moco_train_mode_encoder_k = False

        _metrics = {}
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Prediction step
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs,
                                update_kencoder_queue=False,
                                report_align_unif=True,
                                report_metrics=True)
            for k, v in outputs['specific_losses'].items():
                if isinstance(v, torch.Tensor):
                    v = v.detach()
                    v = self._nested_gather(v)
                else:
                    _v = torch.Tensor(1)
                    _v[0] = v
                    v = _v
                _metrics[k] = v if k not in _metrics else torch.cat((_metrics[k], v), dim=0)

        metrics = {}
        metric_key_prefix = 'test'  # only test_/eval_ are accepted by HF trainers
        for k, v in _metrics.items():
            metrics[f"{metric_key_prefix}_{k}"] = v.float().mean().item()

        model.moco_train_mode_encoder_k = True
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=len(dataloader))

    def evaluate_beir(self, epoch, output_dir, sim_function, batch_size=32, beir_datasets=None) -> Dict[str, float]:
        # clear cache to prevent job from running into errors with multi-GPU on large datasets (likely to be related to GPU memory)
        #       ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: -9) local_rank: 0 (pid: 106555) of binary:
        #       16gpu will error out on datasets >= NQ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not beir_datasets:
            # fever will cause gpu error when `Encoding Batch 88/109`
            # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact'] # quick test
            # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'quora', 'dbpedia-entity', 'nq'] # mostly reported in Contriever
            # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'trec-covid', 'nq', 'dbpedia-entity', 'quora'] # small testsets+NQ+FEVER+Quora
            beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'trec-covid', 'quora', 'nq']  # smallest 8 datasets+quora,nq
            # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'trec-covid']  # smallest 8 datasets
            # beir_datasets = ['fiqa']  # test
        norm_query = self.model.norm_query
        norm_doc = self.model.norm_doc
        beir_data_path = self.training_args.beir_path

        metrics = {'epoch': epoch}
        avg_ndcg_10 = []
        avg_recall_100 = []

        for dataset in beir_datasets:
            logger.info(f"Start evaluating with dataset={dataset}")
            split = 'dev' if dataset == 'msmarco' else 'test'
            ndcg, _map, recall, precision, mrr, recall_cap, hole = beir_utils.evaluate_model(
                query_encoder=self.model,
                doc_encoder=self.model,
                tokenizer=self.tokenizer,
                dataset=dataset,
                batch_size=batch_size,
                norm_query=norm_query,
                norm_doc=norm_doc,
                is_main=dist_utils.is_main(),
                split=split,
                metric=sim_function,
                beir_data_path=beir_data_path,
                add_qd_prompt=(self.training_args.dq_prompt_ratio > 0.0)
            )

            if dist_utils.is_main():
                # logger.info(dataset + ' ' + str(ndcg))
                # logger.info(dataset + ' ' + str(_map))
                # logger.info(dataset + ' ' + str(recall))
                # logger.info(dataset + ' ' + str(precision))
                # logger.info(dataset + ' ' + str(mrr))
                # logger.info(dataset + ' ' + str(recall_cap))
                # logger.info(dataset + ' ' + str(hole))

                metrics[f'eval_{dataset}_ndcg@10'] = ndcg['NDCG@10']
                metrics[f'eval_{dataset}_recall@100'] = recall['Recall@100']
                avg_ndcg_10.append(ndcg['NDCG@10'])
                avg_recall_100.append(recall['Recall@100'])

                result_dict = {
                    'dataset': dataset,
                    'split': split,
                    'metric': sim_function,
                    'norm_query': norm_query,
                    'norm_doc': norm_doc,
                    'scores': {
                        'ndcg': ndcg,
                        'map': _map,
                        'precision': precision,
                        'recall': recall,
                        'mrr': mrr,
                        'recall_cap': recall_cap,
                        'hole': hole,
                    }
                }
                with open(f"{output_dir}/{dataset}.json", 'w') as writer:
                    writer.write(json.dumps(result_dict, indent=4) + "\n")
                rows = ['metric,@1,@3,@5,@10,@100,@1000']
                for metric_name, scores in result_dict['scores'].items():
                    row = ','.join([str(s) for s in ([metric_name] + list(scores.values()))])
                    rows.append(row)
                with open(f"{output_dir}/{dataset}.csv", 'w') as writer:
                    for row in rows:
                        writer.write(row + "\n")

        metrics['eval_avg_ndcg@10'] = np.mean(avg_ndcg_10)
        metrics['eval_avg_recall@100'] = np.mean(avg_recall_100)

        return metrics

    def evaluate_senteval(
        self, epoch, output_dir, eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']
        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        self.model.eval()
        results = se.eval(tasks)
        
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

        metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2} 
        if eval_senteval_transfer or self.args.eval_transfer:
            avg_transfer = 0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                avg_transfer += results[task]['devacc']
                metrics['eval_{}'.format(task)] = results[task]['devacc']
            avg_transfer /= 7
            metrics['eval_avg_transfer'] = avg_transfer

        results.update(metrics)
        with open(f"{output_dir}/senteval.json", 'w') as writer:
            writer.write(json.dumps(results, indent=4) + "\n")

        return metrics

    def _dump_eval_output(self, eval_output, eval_output_path):
        with open(eval_output_path, 'w') as eval_output_writer:
            print("Dumping eval outputs to %s" % eval_output_path)
            eval_output_writer.write(json.dumps(eval_output) + '\n')

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_sampler = SequentialSampler(eval_dataset)
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0, # self.args.dataloader_num_workers, # it causes collapse with multiple workers
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, extra_logs=None):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if extra_logs:
                for k, v in extra_logs.items():
                    if isinstance(v, torch.Tensor):
                        v = self._nested_gather(v).mean().item()
                    if k in {'queue_ptr', 'active_queue_size'}:
                        logs[k] = round(v)
                    else:
                        logs[k] = round(v / (self.state.global_step - self._globalstep_last_logged), 4)
                    # logs[k] = round(self._nested_gather(v).mean().item() / (self.state.global_step - self._globalstep_last_logged), 4)
                    extra_logs[k] = 0.0

            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = {}
        if self.args.do_eval and self.control.should_evaluate:
            eval_output_dir = os.path.join(self.eval_output_dir, 'epoch-%.2f_step-%d' % (epoch, self.state.global_step))
            os.makedirs(eval_output_dir, exist_ok=True)
            # dev score
            if self.eval_dataset:
                with utils.numpy_seed(self.args.seed):
                    metrics_dev = self.evaluate(self.eval_dataset, metric_key_prefix="dev")
                    metrics.update(metrics_dev)
            # beir-eval
            metrics_beir = self.evaluate_beir(epoch=epoch, output_dir=eval_output_dir,
                                              sim_function=self.model.sim_type,
                                              batch_size=self.training_args.beir_batch_size,
                                              # batch_size=self.args.per_device_eval_batch_size,
                                              beir_datasets=self.training_args.beir_datasets)
            metrics.update(metrics_beir)
            # sent-eval
            # major_metric = 'eval_avg_transfer' if 'eval_avg_transfer' in metrics else 'eval_avg_sts'
            metrics_senteval = self.evaluate_senteval(epoch=epoch, output_dir=eval_output_dir)
            metrics.update(metrics_senteval)
            metrics["step"] = self.state.global_step
            for k, v in metrics.items():
                if isinstance(v, float64):
                    metrics[k] = float(v)
            self.log(metrics)
            major_metric = 'eval_avg_ndcg@10'
            self._report_to_hp_search(trial, epoch, metrics)

            # check if it's a new best score
            major_score = metrics[major_metric]
            metrics['major_score'] = major_score
            self.eval_writer.write(json.dumps(metrics) + '\n')
            eval_output_path = os.path.join(eval_output_dir, 'avg_score.json')
            self._dump_eval_output(metrics, eval_output_path)

            if major_score > self.best_score:
                self.early_stop_counter = 0
                logger.info("New best score: %s=%.6f", major_metric, major_score)
                print("\nNew best score: %s=%.6f (%.2f%%), previous best was %.6f" %
                      (major_metric, major_score,
                       ((major_score - self.best_score) / self.best_score) * 100.0 if self.best_score != 0.0 else 0.0,
                       self.best_score))
                self.best_score = major_score
                self.save_model(os.path.join(self.args.output_dir, 'best_ckpt'))
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    self.control.should_training_stop = True

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        # do not use, will significantly slow down the training
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """

        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save.
        assert _model_unwrap(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        output_dir = os.path.join(self.args.output_dir, 'checkpoints', checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self, 'model_data_training_args'):
            torch.save(self.model_data_training_args, os.path.join(output_dir, "model_data_training_args.bin"))

        self.store_flos()

        self.save_model(output_dir)
        if self.deepspeed:
            self.deepspeed.save_checkpoint(output_dir)

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)
        elif self.is_world_process_zero() and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            reissue_pt_warnings(caught_warnings)

        # Save the Trainer state
        if self.is_world_process_zero():
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Maybe delete some older checkpoints.
        if self.is_world_process_zero():
            self._rotate_checkpoints(use_mtime=True)


    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = self.get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
            )
        return self.lr_scheduler

    def get_scheduler(self, name, optimizer):
        """
        Modified based on optimization.get_scheduler()
        """
        num_warmup_steps = self.args.get_warmup_steps(self.args.max_steps)
        num_training_steps = self.args.max_steps
        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
        if name == SchedulerType.CONSTANT:
            return schedule_func(optimizer)
        elif name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
        elif name == SchedulerType.COSINE:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps, num_cycles=self.args.num_cycles)
        elif name == SchedulerType.COSINE_WITH_RESTARTS:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps, num_cycles=self.args.num_cycles)
        elif name == SchedulerType.COSINE_WITH_DECAYED_RESTARTS:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps, num_cycles=self.args.num_cycles)
        elif name == SchedulerType.POLYNOMIAL:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps,
                                 lr_end=self.args.lr_end, power=self.args.power)

        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], forward_only: bool=False) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        extra_losses = {}

        with self.autocast_smart_context_manager():
            if forward_only:
                with torch.no_grad():
                    _, _ = self.compute_loss(model, inputs, return_outputs=True)
                    model.zero_grad()
                    return None, None
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            if hasattr(outputs, 'specific_losses'):
                # print('batch_size=', outputs['specific_losses']['batch_size'])
                for k, v in outputs.specific_losses.items():
                    extra_losses[k] = v
                sent_lens = inputs['length'].type(torch.float).mean(dim=0)
                for li, l in enumerate(sent_lens):
                    extra_losses[f'sent_len_{li}'] = l
                    # print(f'sent_len_{li}={l}')

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            for k, v in extra_losses.items():
                extra_losses[k] = v / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach(), extra_losses


    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        
        The main difference between ours and Huggingface's original implementation is that we 
        also load model_args when reloading best checkpoints for evaluation.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self.eval_output_dir = os.path.join(self.args.output_dir, 'eval_output')
        os.makedirs(self.eval_output_dir, exist_ok=True)
        output_eval_file = os.path.join(self.args.output_dir, f'eval_results.json')
        self.eval_writer = open(output_eval_file, 'w')

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)
            if not self.is_model_parallel:
                model = model.to(self.args.device)

            self.model = model
            self.model_wrapped = model

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )

        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
            self.model = model.module
            self.model_wrapped = model  # will get further wrapped in DDP
            self.deepspeed = model  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        model = self.model_wrapped

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            print('Initializing DistributedDataParallel')
            print('self.args.local_rank=', self.args.local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        if self.args.local_rank == 0 or self.args.local_rank == -1:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        extra_logs = collections.defaultdict(float)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(train_dataloader) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            # assert train_dataset_is_sized, "currently we only support sized dataloader!"
            # prepare for the warmup for queue size
            if self.model.warmup_queue_size_ratio > 0:
                fullsize_step = int(self.model.warmup_queue_size_ratio * self.args.max_steps)
                num_warmup_stage = int(self.model.queue_size / total_train_batch_size)
                warmup_steps = {(fullsize_step // (num_warmup_stage - 1) * i): i+1 for i in range(1, num_warmup_stage)}
                self.model.active_queue_size = total_train_batch_size
            else:
                self.model.active_queue_size = self.model.queue_size

            inputs = None
            last_inputs = None
            for step, inputs in enumerate(epoch_iterator):
                if self.model.warmup_queue_size_ratio > 0 and self.state.global_step in warmup_steps:
                    self.model.active_queue_size = total_train_batch_size * warmup_steps[self.state.global_step]
                    print(f'step={(step + 1)}, model.active_queue_size={self.model.active_queue_size}')
                inputs['report_metrics'] = True if (step + 1) % self.args.logging_steps == 0 else False
                # Skip any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                if (step + 1) % self.model.queue_update_steps != 0:
                    # only feedforward the model to update the queue
                    _, _ = self.training_step(model, inputs, forward_only=True)
                    # self.state.global_step += 1
                    # self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    # steps_trained_progress_bar.update(1)
                    continue
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)
                if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        loss, extra_losses = self.training_step(model, inputs)
                        tr_loss += loss
                        for k, v in extra_losses.items():
                            extra_logs[k] += v
                else:
                    loss, extra_losses = self.training_step(model, inputs)
                    tr_loss += loss
                    for k, v in extra_losses.items():
                        extra_logs[k] += v
                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping
                        if self.use_cuda_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)
                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )
                    # Optimizer step
                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_cuda_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, extra_logs=extra_logs)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, extra_logs=extra_logs)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint, model_args=self.model_args)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)