import torch

from data_generator import make_domain_profile, sample_unpaired_batch, sample_paired_batch
from data_preprocessing import RawPreprocessor, channel_brightness_vector
from model import MERIT
from train import TrainConfig, train_merit
from evaluate import evaluate_translation, plot_translation_result


def select_reference_by_brightness(query: torch.Tensor, ref_pool: torch.Tensor) -> torch.Tensor:
    """Brightness-based reference selection (Sec. 3, 'Training Details').

    For a single query image, pick the image in `ref_pool` whose per-channel
    (RGGB) mean brightness vector is closest (L2) to the query's.

    Args:
        query:    (4, H, W) or (1, 4, H, W) linearized RAW image.
        ref_pool: (N, 4, H, W) candidate reference images from the target domain.

    Returns:
        The single best-matching reference image, shape (1, 4, H, W).
    """
    if query.dim() == 3:
        query = query.unsqueeze(0)

    query_brightness = channel_brightness_vector(query)          # (1, 4)
    pool_brightness = channel_brightness_vector(ref_pool)        # (N, 4)

    dist = torch.cdist(query_brightness, pool_brightness)        # (1, N)
    best_idx = torch.argmin(dist, dim=1).item()
    return ref_pool[best_idx: best_idx + 1]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    profile_a = make_domain_profile("CameraA", seed=0)
    profile_b = make_domain_profile("CameraB", seed=1)
    preprocessor = RawPreprocessor(black_level=0.02, white_level=1.0)

    model = MERIT(in_ch=4)
    print(model)
    cfg = TrainConfig(steps=150, batch_size=8, img_size=128, device=device)
    model = train_merit(model, profile_a, profile_b, preprocessor, cfg)

    model.eval()
    with torch.no_grad():
        query_raw, gt_raw = sample_paired_batch(1, profile_a, profile_b, size=cfg.img_size)
        query, ground_truth = preprocessor.linearize(query_raw.to(device)), preprocessor.linearize(gt_raw.to(device))

        ref_pool_raw = sample_unpaired_batch(16, profile_b, size=cfg.img_size).to(device)
        ref_pool = preprocessor.linearize(ref_pool_raw)
        reference = select_reference_by_brightness(query[0], ref_pool)

        translated = model.translate(query, reference)

        print("\n--- Inference summary ---")
        print(f"Query (A) mean brightness / channel:      {channel_brightness_vector(query)[0].tolist()}")
        print(f"Selected ref (B) mean brightness / channel:{channel_brightness_vector(reference)[0].tolist()}")
        print(f"Output   (B) mean brightness / channel:    {channel_brightness_vector(translated)[0].tolist()}")

        metrics = evaluate_translation(translated, ground_truth)
        print("\n--- Evaluation vs. paired ground truth ---")
        print(metrics)

        save_path = plot_translation_result(
            source=query, reference=reference, translated=translated,
            target=ground_truth, metrics=metrics, save_path="translation_result.png",
        )
        print(f"\nSaved qualitative comparison to: {save_path}")


if __name__ == "__main__":
    main()