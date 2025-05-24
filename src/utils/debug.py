import torch

def compare_weights(suffix):
    before = torch.load(f"checkpoints/before_{suffix}.pt")
    after = torch.load(f"checkpoints/after_{suffix}.pt")

    for k in before:
        diff = torch.norm(before[k] - after[k]).item()
        print(f"{k}: Δ={diff:.6f}")
