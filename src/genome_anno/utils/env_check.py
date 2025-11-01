from __future__ import annotations
import platform
import torch

def main() -> None:
    print("Python:", platform.python_version())
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA build:", torch.version.cuda)
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    main()
