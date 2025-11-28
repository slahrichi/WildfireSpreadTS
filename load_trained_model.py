import argparse
import torch

from src.models.SMPModel import SMPModel
from src.models.SMPTempModel import SMPTempModel


def infer_in_channels_from_state(state_dict: dict) -> int:
    """
    Infer input channel count C from the conv1 weight in the checkpoint.
    We look for keys ending in 'encoder.conv1.weight' first, then any 'conv1.weight'.
    """
    candidates = [k for k in state_dict.keys() if k.endswith("encoder.conv1.weight")]
    if not candidates:
        candidates = [k for k in state_dict.keys() if k.endswith("conv1.weight")]

    if not candidates:
        raise RuntimeError(
            "Could not find a conv1 weight in state_dict to infer in_channels."
        )

    key = candidates[0]
    weight = state_dict[key]
    if weight.ndim != 4:
        raise RuntimeError(f"Unexpected conv1 weight shape for key {key}: {weight.shape}")
    in_ch = weight.shape[1]
    print(f"Inferred n_channels={in_ch} from checkpoint key '{key}'")
    return in_ch


def build_model(args, n_channels: int):
    """
    Instantiate UNet (SMPModel) or UTAE (SMPTempModel)
    with automatic flatten_temporal_dimension logic and given n_channels.
    """
    if args.model == "unet":
        flatten_temporal_dimension = True
    elif args.model == "utae":
        flatten_temporal_dimension = False
    else:
        raise ValueError(f"Unknown model: {args.model}")

    common_kwargs = dict(
        encoder_name=args.encoder_name,
        n_channels=n_channels,
        flatten_temporal_dimension=flatten_temporal_dimension,
        pos_class_weight=args.pos_class_weight,
        encoder_weights=args.encoder_weights,
        loss_function=args.loss_function,
    )

    if args.model == "unet":
        return SMPModel(**common_kwargs)
    else:
        return SMPTempModel(**common_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained SMP/SMPTemp model and run a dummy forward pass."
    )

    parser.add_argument("--weights_path", required=True)
    parser.add_argument(
        "--model",
        choices=["unet", "utae"],
        default="unet",
        help="Use 'unet' (SMPModel) or 'utae' (SMPTempModel).",
    )

    # Optional: only used for sanity-checking against inferred n_channels
    parser.add_argument(
        "--feature-set",
        help="Feature set: Veg, Multi, or All (case-insensitive). Used for checks only.",
    )
    parser.add_argument(
        "--T",
        type=int,
        choices=[1, 5],
        help="Number of temporal steps used during training. Used for checks only.",
    )

    # Hyperparameters
    parser.add_argument("--encoder-name", default="resnet18")
    parser.add_argument("--pos-class-weight", type=float, default=1.0)
    parser.add_argument("--loss-function", default="Dice")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    # --- load raw state first ---
    raw_state = torch.load(args.weights_path, map_location=args.device)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state = raw_state["state_dict"]
    else:
        state = raw_state

    # --- infer n_channels from checkpoint ---
    inferred_n_channels = infer_in_channels_from_state(state)

    # --- build model using inferred channels ---
    model = build_model(args, inferred_n_channels)

    # --- load weights into model ---
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    model.eval()

    print(f"\nLoaded weights from {args.weights_path}")
    print(f"Model: {args.model}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Inferred n_channels: {model.hparams.n_channels}")
    print(f"flatten_temporal_dimension: {model.hparams.flatten_temporal_dimension}\n")

    # --- dummy inference ---
    if args.model == "unet":
        # SMPModel: expects [B, C, H, W]
        dummy = torch.randn(1, model.hparams.n_channels, 256, 256, device=args.device)
    else:
        # SMPTempModel (UTAE): expects [B, T, C, H, W]
        # Use args.T if provided, else default to 5 just for this sanity check
        T_dummy = args.T if args.T is not None else 5
        dummy = torch.randn(1, T_dummy, model.hparams.n_channels, 256, 256, device=args.device)

    with torch.no_grad():
        logits = model(dummy)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    print("logits:", logits.shape)
    print("probs:", probs.shape)
    print("preds:", preds.shape)
    
if __name__ == "__main__":
    main()
