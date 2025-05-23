import torch
from accelerate import Accelerator

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

try:
    import bitsandbytes as bnb
    print("bitsandbytes installed:", bnb.__version__)
except ImportError:
    print("bitsandbytes not installed")

accelerator = Accelerator()
print("Accelerator initialized successfully!")