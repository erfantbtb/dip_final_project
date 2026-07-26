from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data_generator import DomainProfile, sample_unpaired_batch
from data_preprocessing import RawPreprocessor
from model import MERIT


@dataclass
class TrainConfig:
    steps: int = 150
    batch_size: int = 8
    img_size: int = 128
    lr: float = 2e-4
    lambda_cycle: float = 10.0
    lambda_style: float = 1.0
    log_every: int = 25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_merit(
        model: MERIT, 
        profile_a: DomainProfile, 
        profile_b: DomainProfile,
        preprocessor: RawPreprocessor, 
        cfg: TrainConfig
        ) -> MERIT:
    
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    model.train()
    for step in range(1, cfg.steps + 1):
        raw_a = sample_unpaired_batch(cfg.batch_size, profile_a, size=cfg.img_size).to(cfg.device)
        raw_b = sample_unpaired_batch(cfg.batch_size, profile_b, size=cfg.img_size).to(cfg.device)
        img_a = preprocessor.linearize(raw_a)
        img_b = preprocessor.linearize(raw_b)

        style_b = model.style_encoder(img_b)
        fake_b = model.generator(img_a, style_b)

        style_fake_b = model.style_encoder(fake_b)
        style_loss = F.l1_loss(style_fake_b, style_b)

        style_a = model.style_encoder(img_a)
        rec_a = model.generator(fake_b, style_a)
        cycle_loss = F.l1_loss(rec_a, img_a)

        loss = cfg.lambda_cycle * cycle_loss + cfg.lambda_style * style_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.log_every == 0 or step == 1:
            print(f"[step {step:4d}/{cfg.steps}] "
                  f"total={loss.item():.4f}  cycle={cycle_loss.item():.4f}  "
                  f"style={style_loss.item():.4f}")

    return model