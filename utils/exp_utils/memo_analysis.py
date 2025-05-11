import logging
import socket
from datetime import datetime
import psutil

import torch


# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def print_memory_usage():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024 ** 3):.6f} GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024 ** 3):.6f} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.6f} GB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / (1024 ** 3):.6f} GB\n")


def print_RAM_usage():
    mem_info = psutil.virtual_memory()
    total_memory = mem_info.total / 1024 ** 3  # GB
    available_memory = mem_info.available / 1024 ** 3  # GB

    print("=" * 20)
    print(f"Total RAM: {total_memory:.2f} GB")
    print(f"Available RAM: {available_memory:.2f} GB")
    print("=" * 20, '\n')


def set_up_logger():
    logging.basicConfig(
       format="%(levelname)s:%(asctime)s %(message)s",
       level=logging.INFO,
       datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    return logger


def start_record_memory_history(logger):
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return logger

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)
    return logger


def export_memory_snapshot(logger):
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return logger

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
        return logger
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return logger


def stop_record_memory_history(logger):
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")

    else:
        logger.info("Stopping snapshot record_memory_history")
        torch.cuda.memory._record_memory_history(enabled=None)


def count_params(model: torch.nn.Module):
    # Only count trainable params with requires_grad=False
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params



