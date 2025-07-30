import datetime
from functools import partial
import os

from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import orbax
from orbax.checkpoint import CheckpointManager, checkpoint_utils
from jax import tree_util
import optax
from typing import TypeVar, Mapping
from ml_collections import config_flags, ConfigDict
import optax
import tensorflow as tf
import tqdm
import wandb
import json

from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
from easydict import EasyDict as edict

from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text,
    Timer,
    TrainState,
)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass


TX = TypeVar("TX", bound=optax.OptState)

def restore_optimizer_state(opt_state: TX, restored) -> TX:
    """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
    return tree_util.tree_unflatten(
        tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
    )

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_string("experiment_path", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_config.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    initialize_compilation_cache()
    devices = jax.devices()
    with open(tf.io.gfile.join(FLAGS.experiment_path, 'finetune_config.json'), 'r') as f:
        config = edict(json.load(f))
    logging.info(
        f"""
        Continue Octo Finetuning Script
        ======================
        Pretrained model: {config.pretrained_path}
        Finetuning Dataset: {config.dataset_kwargs.name}
        Data dir: {config.dataset_kwargs.data_dir}
        Task Modality: {config.modality}
        Finetuning Mode: {config.finetuning_mode}

        # Devices: {jax.device_count()}
        Batch size: {config.batch_size} ({config.batch_size // len(devices) } per device)
        # Steps: {config.num_steps}
    """
    )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    assert (
        config.batch_size % len(devices) == 0
    ), f"Batch size ({config.batch_size}) must be divisible by the number of devices ({len(devices)})"
    assert (
        config.viz_kwargs.eval_batch_size % len(devices) == 0
    ), f"Eval batch size ({config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({len(devices)})"

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #########
    #
    # Setup WandB
    #
    #########

    name = format_name_with_config(
        FLAGS.name,
        config,
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        config=config,
        id=wandb_id,
        name=name,
        mode="disabled" if FLAGS.debug else None,
        **config.wandb,
    )

    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########

    model = OctoModel.load_pretrained(
        FLAGS.experiment_path,
    )

    #########
    #
    # Setup Data Loader
    #
    #########

    # create text processor
    if model.config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(model.config["text_processor"])()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    dataset = make_single_dataset(
        config.dataset_kwargs,
        traj_transform_kwargs=config.traj_transform_kwargs,
        frame_transform_kwargs=config.frame_transform_kwargs,
        train=True,
        split=config.get('split', None),
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(config.shuffle_buffer_size)
        .batch(config.batch_size)
        .iterator()
    )
    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    #########
    #
    # Load Pretrained Model
    #
    #########


    #########
    #
    # Setup Optimizer and Train State
    #
    #########

    state_checkpointer = orbax.checkpoint.CheckpointManager(
                tf.io.gfile.join(FLAGS.experiment_path, 'state'),
                orbax.checkpoint.PyTreeCheckpointer(),
                options=orbax.checkpoint.CheckpointManagerOptions(
                    max_to_keep=1,
                ),
            )
    start_step = state_checkpointer.latest_step() + 1
    state_restored = state_checkpointer.restore(step=state_checkpointer.latest_step())
    
    tx, lr_callable, param_norm_callable = create_optimizer(
        state_restored['model']['params'],
        **config.optimizer,
    )

    opt_state_example = tx.init(model.params)
    opt_state = restore_optimizer_state(opt_state_example, state_restored['opt_state'])

    train_state = TrainState(rng=state_restored['rng'],
           model=model,
           step=state_restored['step'],
           opt_state=opt_state,
           tx=tx
           )

    #########
    #
    # Save all metadata
    #
    #########
    save_dir = FLAGS.experiment_path
    
    wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
    logging.info("Saving to %s", save_dir)
    save_callback = SaveCallback(save_dir)

    # Add window_size to top of config, to make eval easier
    new_config = ConfigDict(model.config)
    new_config["window_size"] = example_batch["observation"][
        "timestep_pad_mask"
    ].shape[1]
    model = model.replace(config=new_config)

    # example_batch_spec = jax.tree_map(
    #     lambda arr: (arr.shape, str(arr.dtype)), example_batch
    # )
    # wandb.config.update(
    #     dict(example_batch_spec=example_batch_spec), allow_val_change=True
    # )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # action head knows to pull out the "action" readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    # Data parallelism
    # Model is replicated across devices, data is split across devices
    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    #########
    #
    # Build validation & visualization callbacks
    #
    #########

    if config.modality == "image_conditioned":
        modes_to_evaluate = ["image_conditioned"]
    elif config.modality == "text_conditioned":
        modes_to_evaluate = ["text_conditioned"]
    elif config.modality == "multimodal":
        modes_to_evaluate = ["image_conditioned", "text_conditioned"]
    else:
        modes_to_evaluate = ["base"]

    dataset_kwargs_list = [config.dataset_kwargs]

    val_callback = ValidationCallback(
        loss_fn=loss_fn,
        process_batch_fn=process_batch,
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=config,
        modes_to_evaluate=modes_to_evaluate,
        val_split=config.get('val_split', None),
        **config.val_kwargs,
    )

    viz_callback = VisualizationCallback(
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=config,
        modes_to_evaluate=modes_to_evaluate,
        val_split=config.get('val_split', None),
        **config.viz_kwargs,
    )

    #########
    #
    # Optionally build visualizers for sim env evals
    #
    #########

    if "rollout_kwargs" in config:
        rollout_callback = RolloutVisualizationCallback(
            text_processor=text_processor,
            unnormalization_statistics=dataset.dataset_statistics["action"],
            **config.rollout_kwargs,
        )
    else:
        rollout_callback = None

    #########
    #
    # Train loop
    #
    #########

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)
    
    def save_data(data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    
    with open(tf.io.gfile.join(FLAGS.experiment_path, 'update_info.json'), 'r') as f:
        update_info_list = json.load(f)
    update_info_list = update_info_list[:start_step]

    logging.info(f"Starting from {start_step} step")
    timer = Timer()
    for i in tqdm.tqdm(
        range(start_step, int(config.num_steps)),
        total=int(config.num_steps - start_step),
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )
            update_info = {key: float(val) for key, val in update_info.items()}
            update_info_list.append(update_info)
            save_data(update_info_list, save_dir + '/update_info.json')

        if (i + 1) % config.eval_interval == 0:
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

            with timer("visualize"):
                viz_metrics = viz_callback(train_state, i + 1)
                wandb_log(viz_metrics, step=i)

            if rollout_callback is not None:
                with timer("rollout"):
                    rollout_metrics = rollout_callback(train_state, i + 1)
                    wandb_log(rollout_metrics, step=i)

        if (i + 1) % config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    app.run(main)
