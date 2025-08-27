import torch
from typing import Iterable, Set


def print_module_tensors_and_constants(ts_path: str) -> bool:
    float_64_flag = False
    m = torch.jit.load(ts_path, map_location="cpu")
    print(f"Loaded TorchScript: {ts_path}")
    print(f"Current PyTorch default dtype: {torch.get_default_dtype()}\n")

    # ---- Parameters ----
    print("=== PARAMETERS (name -> dtype, shape) ===")
    has_params = False
    for name, p in m.named_parameters(recurse=True):
        has_params = True
        print(f"{name:60s} {str(p.dtype):12s} {tuple(p.shape)}")
        if p.dtype == torch.float64:
            float_64_flag = True
    if not has_params:
        print("(none)")
    print()

    # ---- Buffers ----
    print("=== BUFFERS (name -> dtype, shape) ===")
    has_buffers = False
    for name, b in m.named_buffers(recurse=True):
        has_buffers = True
        print(f"{name:60s} {str(b.dtype):12s} {tuple(b.shape)}")
        if b.dtype == torch.float64:
            float_64_flag = True
    if not has_buffers:
        print("(none)")
    print()
    
    return float_64_flag


def cast_jit_to_float32(input_path: str, output_path: str):
    """
    Load a TorchScript .pt file, cast all parameters/buffers/constants to float32,
    and save to a new file.
    """
    m = torch.jit.load(input_path, map_location="cpu")

    # ---- Parameters & Buffers ----
    for name, p in m.named_parameters(recurse=True):
        if p.dtype == torch.float64:
            with torch.no_grad():
                setattr(m, name.replace('.', '_'), None)  # break ref
            p.data = p.data.float()
    for name, b in m.named_buffers(recurse=True):
        if b.dtype == torch.float64:
            with torch.no_grad():
                setattr(m, name.replace('.', '_'), None)
            b.data = b.data.float()

    # ---- Submodules ----
    for sub in m.modules():
        for name, p in sub._parameters.items():
            if p is not None and p.dtype == torch.float64:
                sub._parameters[name] = p.float()
        for name, b in sub._buffers.items():
            if b is not None and b.dtype == torch.float64:
                sub._buffers[name] = b.float()

    # Save back
    m = m.to(torch.float32)
    m.save(output_path)
    print(f"Saved float32-converted model → {output_path}")

if __name__ == "__main__":
    # Usage: set your file path here
    float_64_flag = print_module_tensors_and_constants("0816_171833-CLN025-jit.pt")
    if float_64_flag:
        cast_jit_to_float32("0816_171833-CLN025-jit.pt", "0816_171833-CLN025-f32-jit-f32.pt")