import os
import pathlib
import tempfile
import functools
import argparse
 
import torch
os.environ["PT_HPU_LAZY_MODE"] = "0"
from torch.utils.data import DataLoader
 
import habana_frameworks.torch.core as htcore
import transformers
import datasets
import tqdm
import torch.nn.functional as F
import instructlab.training.data_process as dp
import numpy as np
from copy import deepcopy
 
from config import DataProcessArgs
from utils import retrieve_chat_template
from tokenizer_utils import setup_tokenizer
from token_dataset import setup_dataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DistributedSampler
from transformers import PreTrainedModel, get_scheduler
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from instructlab.training.utils import get_module_class_from_name
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from optimum.habana.accelerate import GaudiAccelerator
import habana_frameworks.torch.core as htcore

import habana_frameworks.torch.distributed.hccl
 
 
device_hpu = torch.device('hpu')
 
MODEL = "instructlab/granite-7b-lab"
_TMP_DIR = tempfile.TemporaryDirectory()
TMP_DIR = pathlib.Path(_TMP_DIR.name)
TMP_DATA_DIR = TMP_DIR / "data"
TMP_PREPARED_DATA_PATH = TMP_DATA_DIR / "data.jsonl"
TMP_CHECKPOINT_DIR = TMP_DIR / "checkpoints"
NUM_GPUS=1
 
CHAT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "chat_templates/ibm_generic_tmpl.py")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # initialize the process group
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    os.environ['ACCELERATE_USE_FSDP'] = "True"
    os.environ['FSDP_CPU_RAM_EFFICIENT_LOADING'] = "True"


def cleanup():
    dist.destroy_process_group()

def get_fsdp_config(model: PreTrainedModel, sharding_strategy, cpu_offload):
    # Third Party
    from accelerate.utils import FullyShardedDataParallelPlugin
    from optimum.habana.accelerate.utils import GaudiFullyShardedDataParallelPlugin
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

    block_name = model._no_split_modules[0]

    wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                get_module_class_from_name(model, block_name),
            },
    )
    prefetch_policy = (
        BackwardPrefetch.BACKWARD_PRE
    )

    fsdp_plugin = GaudiFullyShardedDataParallelPlugin(
            auto_wrap_policy=wrap_policy,
            limit_all_gathers=True,
            mixed_precision_policy=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            backward_prefetch=prefetch_policy,
            sharding_strategy=ShardingStrategy[sharding_strategy],
            cpu_offload=CPUOffload(cpu_offload),
            #state_dict_type = StateDictType.FULL_STATE_DICT if device.is_hpu() else None,
    )
    return fsdp_plugin

def setup_accelerator(model: PreTrainedModel, sharding_strategy, cpu_offload):
    accel_args = {
        "fsdp_plugin": get_fsdp_config(model, sharding_strategy, cpu_offload)
    }
    accelerator = GaudiAccelerator(
        **accel_args,
    )
    return accelerator


def pad_collate_fn(batch, pad_token_id, max_pad_len):
    lens = np.array([len(item["input_ids"]) for item in batch])
    max_len = max_pad_len
 
    input_ids = torch.stack(
        [
            F.pad(
                item["input_ids"],
                (max_len - len(item["input_ids"]), 0),
                mode="constant",
                value=pad_token_id,
            )
            for item in batch
        ]
    )
    labels = torch.stack(
        [
            F.pad(
                item["labels"],
                (max_len - len(item["labels"]), 0),
                mode="constant",
                value=-100,
            )
            for item in batch
        ]
    )
    num_loss_counted_tokens = (labels != -100).sum()
 
    attention_mask = torch.stack(
        [
            F.pad(
                item["attention_mask"],
                (max_len - len(item["attention_mask"]), 0),
                mode="constant",
                value=0,
            )
            for item in batch
        ]
    )
    print(
        f"\033[96m total tokens: {max_len * len(batch)} num samples: {len(batch)} num padding tokens: {max_len * len(batch) - lens.sum()}"
        f"max len: {max_len} min len: {min(lens)} avg len: {lens.mean()} "
        f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
    )
 
    return {
        "input_ids": input_ids,
        "labels": labels,
        "num_loss_counted_tokens": num_loss_counted_tokens,
        "attention_mask": attention_mask,
        "num_samples": len(batch),
    }
 
 
def _tempdir_setup():
    """mkdirs temp directories"""
    TMP_DATA_DIR.mkdir()
    TMP_CHECKPOINT_DIR.mkdir()
 
 
def _prepare_training_data(
    model_path: str,
    data_path: str,
    data_output_path: str,
    max_seq_len: int,
):
    """
    Takes input .jsonl file, renders it in model chat templates,
    tokenizes it, and saves it as a data.jsonl file at TMP_PREPARED_DATA_PATH.
    """
    dp.main(
        DataProcessArgs(
            # XXX(osilkin): make a decision here, either:
            #   1. the CLI is fully responsible for managing where the data is written
            #   2. we never cache it and simply write it to a tmp file every time.
            #
            # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux
            # where the user has a defined place for new temporary data to be written.
            data_output_path=data_output_path,
            model_path=model_path,
            data_path=data_path,
            max_seq_len=max_seq_len,
            chat_tmpl_path=str(CHAT_TEMPLATE_PATH),
        )
    )
 
 
def _prepare_tokenizer(model: str):
    """
    Sets up tokenizer for model, given the model's chat template and
    available special tokens.
    """
    chat_template, special_tokens = retrieve_chat_template(CHAT_TEMPLATE_PATH)
    tokenizer = setup_tokenizer(model, special_tokens, chat_template)
    return tokenizer
 
 
def fsdp_main(
    rank,
    world_size,
    args
):
    setup(rank, world_size)

    data_file = "./sample-data/train_all_pruned_SDG.jsonl"
    max_batch_len = 10000
    is_padding_free = False
    seed = 42
    
    epochs = args.epochs
    max_seq_len = args.max_seq_len
    effective_batch_size = args.batch_size

    _tempdir_setup()
    _prepare_training_data(
        model_path=str(MODEL),
        data_path=data_file,
        data_output_path=str(TMP_DATA_DIR),
        max_seq_len=max_seq_len
    )
 
    tokenizer = _prepare_tokenizer(MODEL)
 
    dataset = setup_dataset(
        str(TMP_PREPARED_DATA_PATH),
    )
 
    #  Get global max_seq_len in dataset
    global_max_len = 0
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        )
    
    for sample in dataloader:
        global_max_len = max(global_max_len, sample["input_ids"].size()[1])
    
    print(f"Global max length: {global_max_len}")

    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
 
    dataloader = DataLoader(
        dataset, 
        batch_size=effective_batch_size, 
        sampler=sampler, 
        collate_fn=functools.partial(
            pad_collate_fn, 
            pad_token_id=tokenizer.pad_token_id,
            max_pad_len=global_max_len,
            )
        )
 
    #sharding_strategy= "SHARD_GRAD_OP"
    sharding_strategy= "FULL_SHARD"
    cpu_offload = True
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    model = model.to(device_hpu)
    model = model.to(dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    accelerator = setup_accelerator(model, sharding_strategy, cpu_offload)
    
    #model = FSDP(model)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=20,
        num_training_steps=epochs * len(dataloader) // args.batch_size,
    )
    model, optimizer, _, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        deepcopy(dataloader),
        lr_scheduler,
    )
    lr_scheduler.split_batches = True
    samples_seen = 0
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=10, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/granite-7b'),
        record_shapes=True, profile_memory=True, with_stack=True
    ) 

    prof.start()
    model.train()

    for epoch in range(epochs):
        ddp_loss = torch.zeros(2).to(device_hpu)
        print(f"IN EPOCH: {epoch}")
        dataloader.sampler.set_epoch(epoch)
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device_hpu)
 
            #-----------------------------------------
            num_loss_counted_tokens = float(
                torch.tensor([batch.pop("num_loss_counted_tokens")])
            )
            micro_batch_size = float(torch.tensor([batch.pop("num_samples")]))
            #-----------------------------------------

            output = model(
               **batch,
               use_cache=False,
            )
 
            loss = output.loss
            #-----------------------------------------
            log_loss = loss.detach().item()

            num_loss_counted_tokens, micro_batch_size, log_loss = map(
                float,
                accelerator.reduce(
                    torch.tensor(
                        [num_loss_counted_tokens, micro_batch_size, log_loss],
                        dtype=torch.float32,
                        device=accelerator.device,
                    ),
                    reduction="sum",
                ),
            )
            samples_seen += int(micro_batch_size)
            #-----------------------------------------
            loss = (
                loss / num_loss_counted_tokens * world_size
            )  
            #loss.backward()
            accelerator.backward(loss)
            htcore.mark_step()
            global_grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #device.mark_step()
            htcore.mark_step()
            prof.step()
            #-----------------------------------------
            optimizer.zero_grad()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += micro_batch_size
 
            if i % 10 == 0:
                print(f"Step {i} Loss: {loss.item()}")
            if i > 0  and i % 33 == 0:
                break

            del loss, output, batch
        
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

    prof.stop()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple FSDP Example')
    parser.add_argument('--world-size', type=int, default=2, help='number of hpus to be used for distributed training')
    parser.add_argument('--batch-size', type=int, default=8, help='effective input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--max-seq-len', type=int, default=512, help='maximum sequence length allowed for training samples')
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(fsdp_main,
        args=(world_size, args),
        nprocs=world_size,
        join=True)
    _TMP_DIR.cleanup()
