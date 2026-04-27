import sys
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import torch
import torchvision
import transformers
import fastai

print("=== Robotics + AI Journey — Day 1 Environment Check ===\n")
print(f"Python:        {sys.version.split()[0]}")
print(f"NumPy:         {np.__version__}")
print(f"Pandas:        {pd.__version__}")
print(f"Matplotlib:    {matplotlib.__version__}")
print(f"Scikit-learn:  {sklearn.__version__}")
print(f"PyTorch:       {torch.__version__}")
print(f"TorchVision:   {torchvision.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"fastai:        {fastai.__version__}")
print(f"\nCUDA available: {torch.cuda.is_available()} (Intel Mac — expected: False)")
print(f"MPS available:  {torch.backends.mps.is_available()} (Apple Silicon — expected: False)")

x = torch.rand(3, 3)
print(f"\nTorch tensor test:\n{x}")
print("\n✓ Alle Imports erfolgreich. Setup abgeschlossen!")
