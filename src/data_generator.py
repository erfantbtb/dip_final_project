from dataclasses import dataclass, field
import torch
from typing import Set 

@dataclass
class DomainProfile:
    name: str
    channel_gain: torch.Tensor     
    read_noise: float               
    shot_noise: float               
    black_level: float = 0.02
    white_level: float = 1.0


def make_domain_profile(name: str, seed: int) -> DomainProfile:
    """This function creates different domains with given seed for reproducability purposes.

    Args:
        name (str): Name of Sensor
        seed (int): Seed number

    Returns:
        DomainProfile: Domain Profile Dataclass 
    """
    g = torch.Generator().manual_seed(seed)
    gain = 0.75 + 0.5 * torch.rand(4, generator=g)
    read_noise = float(0.01 + 0.02 * torch.rand(1, generator=g).item())
    shot_noise = float(0.02 + 0.05 * torch.rand(1, generator=g).item())
    
    domain_profile = DomainProfile(
        name=name, 
        channel_gain=gain,
        read_noise=read_noise, 
        shot_noise=shot_noise
        )
    
    return domain_profile


def _base_scene(batch_size: int, size: int, generator: torch.Generator) -> torch.Tensor:
    """Low-frequency 'scene content' shared as ground truth radiance, so that
    two domains observing the 'same scene' are structurally correlated. Used
    only for the paired sanity-check utility, not for unpaired training."""
    small = torch.rand(batch_size, 4, size // 8, size // 8, generator=generator)
    scene = torch.nn.functional.interpolate(small, 
                                            size=(size, size), 
                                            mode="bilinear",
                                            align_corners=False)
    return scene.clamp(0, 1)


def apply_domain_noise(clean: torch.Tensor, profile: DomainProfile) -> torch.Tensor:
    """
    Apply Poisson-Gaussian-style noise: Var(x) = alpha * z + beta.

    Args:
        clean (torch.Tensor): Clean raw image without sensor specefic noise.
        profile (DomainProfile): Sensor domain profile including noises for generating images.

    Returns:
        torch.Tensor: Sensor specefic generated image
    """
    variance = profile.shot_noise * clean.clamp(min=0) + profile.read_noise ** 2
    noise = torch.randn_like(clean) * variance.sqrt()
    return clean + noise


def render_domain_raw(scene: torch.Tensor, profile: DomainProfile) -> torch.Tensor:
    """
    Turn a clean scene into a domain-specific noisy RAW capture: apply the
    sensor's spectral tint, add sensor noise, then re-embed into [black, white].

    Args:
        scene (torch.Tensor): Raw clean scene
        profile (DomainProfile): Sensor domain profile including noises for generating images.

    Returns:
        torch.Tensor: Sensor Specefic ready data
    """
    gain = profile.channel_gain.view(1, 4, 1, 1).to(scene.device)
    tinted = scene * gain
    noisy = apply_domain_noise(tinted, profile)
    raw = profile.black_level + noisy * (profile.white_level - profile.black_level)
    return raw.clamp(0.0, 1.0)


def sample_unpaired_batch(
        batch_size: int, 
        profile: DomainProfile, 
        size: int = 128,
        generator: torch.Generator = None) -> torch.Tensor:
    """
    Sample a batch of RAW images from a single domain, with independent
    scene content (this is the 'unpaired' setting MERIT trains under).

    Args:
        batch_size (int): _description_
        profile (DomainProfile): _description_
        size (int, optional): _description_. Defaults to 128.
        generator (torch.Generator, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    generator = generator or torch.Generator()
    scene = _base_scene(batch_size, size, generator)
    return render_domain_raw(scene, profile)


def sample_paired_batch(
        batch_size: int, 
        profile_a: DomainProfile, 
        profile_b: DomainProfile,
        size: int = 128, 
        generator: torch.Generator = None) -> Set[torch.Tensor, torch.Tensor]:
    """
    Paired sampling used for evaluation and having ground truth

    Args:
        batch_size (int): batch size
        profile_a (DomainProfile): Sensor a domain profile
        profile_b (DomainProfile): Sensor b domain profile
        size (int, optional): Raw image sizes (height and width). Defaults to 128.
        generator (torch.Generator, optional): Random generator with specefic seed for batch creation. Defaults to None.

    Returns:
        Set[torch.Tensor, torch.Tensor]: _description_
    """
    generator = generator or torch.Generator()
    scene = _base_scene(batch_size, size, generator)
    raw_a = render_domain_raw(scene, profile_a)
    raw_b = render_domain_raw(scene, profile_b)
    return raw_a, raw_b


class RawDomainDataset(torch.utils.data.Dataset):
    """
    Infinite-style dataset: generates a fresh unpaired RAW sample from a
    single domain on every __getitem__ call (RAW data is cheap to synthesize,
    so there's no need to materialize a fixed-size dataset on disk).

    Args:
        torch (_type_): _description_
    """

    def __init__(self, profile: DomainProfile, size: int = 128, length: int = 1000):
        self.profile = profile
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        raw = sample_unpaired_batch(1, self.profile, size=self.size)[0]
        return raw