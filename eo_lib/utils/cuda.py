import gc

import socket
import torch

from numba import cuda
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetCount,
    nvmlDeviceGetUtilizationRates,
    nvmlShutdown,
)

def free_gpu_cache(hard=False, device_id=None):

    print("Initial GPU usage: ", print_gpu_utilization())
    gc.collect()
    torch.cuda.empty_cache()
    if hard:
        cuda.select_device(device_id)
        cuda.close()
        cuda.select_device(device_id)
        _ = torch.cuda.FloatTensor(1)
        raise UserWarning(
            "The context is now closed. "
            "Pytorch manages CUDA initialization internally - "
            "Initializing the device to use again. This may cause weird behaviour"
        )
    print("GPU usage post cleaning", print_gpu_utilization())


def get_gpu_stats():

    gpu_metrics = []
    node_name = socket.gethostname()

    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        util = nvmlDeviceGetUtilizationRates(handle)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        gpu_metrics.append(
            {
                "node": node_name,
                "gpu_id": i,
                "utilization": util.gpu,
                "memory_used": memory_info.used / 1024**2,
                "memory_total": memory_info.total / 1024**2,
            }
        )

    nvmlShutdown()
    return gpu_metrics


def get_device() -> str:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


def get_gpu_count():

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    
    return 0



def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    nvmlShutdown()
