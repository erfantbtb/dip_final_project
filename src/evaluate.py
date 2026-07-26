from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()




@dataclass
class EvalResult:
    mae: float
    psnr: float


    def __str__(self):
        return (f"MAE={self.mae:.4f}  PSNR={self.psnr:.2f} dB")


def evaluate_translation(pred: torch.Tensor, target: torch.Tensor) -> EvalResult:
    return EvalResult(
        mae=compute_mae(pred, target),
        psnr=compute_psnr(pred, target),
    )


def raw_to_preview_rgb(raw: torch.Tensor) -> torch.Tensor:
    squeeze = False
    if raw.dim() == 3:
        raw = raw.unsqueeze(0)
        squeeze = True

    r, gr, gb, b = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    g = 0.5 * (gr + gb)
    rgb = torch.stack([r, g, b], dim=-1).clamp(0, 1)  # (B, H, W, 3)
    return rgb[0] if squeeze else rgb


def plot_translation_result(source: torch.Tensor, reference: torch.Tensor,
                             translated: torch.Tensor, target: torch.Tensor = None,
                             metrics: EvalResult = None, save_path: str = "translation_result.png"):
    panels = [("Source (Domain A)", source), ("Style reference (Domain B)", reference),
              ("Translated (A -> B)", translated)]
    if target is not None:
        panels.append(("Ground truth (Domain B)", target))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, panels):
        preview = raw_to_preview_rgb(img.detach().cpu().squeeze(0) if img.dim() == 4 else img.detach().cpu())
        ax.imshow(preview.numpy())
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    if metrics is not None:
        fig.suptitle(f"Translated vs. ground truth  |  {metrics}", fontsize=10, y=0.02)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path