import torch
import torch.nn.functional as F


class RawPreprocessor:
    def __init__(self, black_level: float = 0.02, white_level: float = 1.0):
        self.black_level = black_level
        self.white_level = white_level

    def linearize(self, raw: torch.Tensor) -> torch.Tensor:
        """
        (raw - black) / (white - black), clipped to [0, 1].

        Args:
            raw (torch.Tensor): Non-Linearized data

        Returns:
            torch.Tensor: Linearized data
        """
        range = max(self.white_level - self.black_level, 1e-6)
        out = (raw - self.black_level) / range
        return out.clamp(0.0, 1.0)

    def unlinearize(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Inverse of `linearize`, useful if you need to re-embed a generated
        image back into raw-sensor units.

        Args:
            raw (torch.Tensor): Linearized Data

        Returns:
            torch.Tensor: Non-Linearized data 
        """
        return self.black_level + raw * (self.white_level - self.black_level)

    @staticmethod
    def pack_bayer(bayer: torch.Tensor) -> torch.Tensor:
        """
        Convert a single-channel mosaiced Bayer image (B, 1, H, W) into a
        4-channel packed RGGB tensor (B, 4, H/2, W/2), following the standard
        `space_to_depth`-style packing used in RAW-to-RAW literature.

        Assumes an RGGB Bayer pattern:
            R  Gr
            Gb B

        Args:
            bayer (torch.Tensor): Single Channeled data

        Returns:
            torch.Tensor: Data with 4 channels
        """
        assert bayer.dim() == 4 and bayer.shape[1] == 1, "expected (B, 1, H, W)"
        r = bayer[:, :, 0::2, 0::2]
        gr = bayer[:, :, 0::2, 1::2]
        gb = bayer[:, :, 1::2, 0::2]
        b = bayer[:, :, 1::2, 1::2]
        return torch.cat([r, gr, gb, b], dim=1)

    @staticmethod
    def unpack_bayer(packed: torch.Tensor) -> torch.Tensor:
        """
        Inverse of `pack_bayer`: (B, 4, H, W) -> (B, 1, 2H, 2W).

        Args:
            packed (torch.Tensor): 4-Channeled Data

        Returns:
            torch.Tensor: Single Channel Data
        """
        assert packed.dim() == 4 and packed.shape[1] == 4, "expected (B, 4, H, W)"
        b, _, h, w = packed.shape
        out = torch.zeros(b, 1, h * 2, w * 2, device=packed.device, dtype=packed.dtype)
        out[:, :, 0::2, 0::2] = packed[:, 0:1]
        out[:, :, 0::2, 1::2] = packed[:, 1:2]
        out[:, :, 1::2, 0::2] = packed[:, 2:3]
        out[:, :, 1::2, 1::2] = packed[:, 3:4]
        return out

    def preprocess(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Full pipeline for already-packed (B, 4, H, W) input: linearize only.
        If a single-channel mosaiced image is passed instead, pack it first.

        Args:
            raw (torch.Tensor): 1-Channeled, non-linear data

        Returns:
            torch.Tensor: 4-channel, linear data
        """
        if raw.shape[1] == 1:
            raw = self.pack_bayer(raw)
        return self.linearize(raw)


def channel_brightness_vector(img: torch.Tensor) -> torch.Tensor:
    """
    Spatial mean per RGGB channel: (B, 4, H, W) -> (B, 4).
    Used by the brightness-based reference selection strategy at inference
    time (Sec. 3, 'Training Details' of the paper).

    Args:
        img (torch.Tensor): Refrence image for choosing target image

    Returns:
        torch.Tensor: Brightness of image
    """
    return img.mean(dim=(2, 3))