import os

from transformers import is_torch_tpu_available


def wandb_setup(cls, args=None, state=None, model=None, model_args=None, training_args=None, moco_args=None, resume=False, **kwargs):
    """
    Modified based on WandbCallback at L534 of transformers.integration
    to keep track of our customized parameters (moodel_args, data_args)
    """
    if cls._wandb is None:
        return
    cls._initialized = True
    if state.is_world_process_zero:
        combined_dict = {**args.to_sanitized_dict()}

        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}
        if model_args is not None:
            model_args = model_args.to_dict()
            combined_dict = {**model_args, **combined_dict}
        if training_args is not None:
            training_args = training_args.to_dict()
            combined_dict = {**training_args, **combined_dict}
        if moco_args is not None:
            moco_args = vars(moco_args)
            combined_dict = {**moco_args, **combined_dict}

        trial_name = state.trial_name
        init_args = {}
        if trial_name is not None:
            run_name = trial_name
            init_args["group"] = args.run_name
        else:
            run_name = args.run_name
        init_args['resume'] = resume

        if cls._wandb.run is None:
            cls._wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"),
                name=run_name,
                **init_args,
            )
        # add config parameters (run may have been created manually)
        cls._wandb.config.update(combined_dict, allow_val_change=True)

        # define default x-axis (for latest wandb versions)
        if getattr(cls._wandb, "define_metric", None):
            cls._wandb.define_metric("train/global_step")
            cls._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

        # keep track of model topology and gradients, unsupported on TPU
        if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
            cls._wandb.watch(
                model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
            )
