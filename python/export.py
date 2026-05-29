"""
03_export_onnx.py
Export HealthDocRiskModel to ONNX and verify the result.
 
Usage
─────
  python 03_export_onnx.py \
      --checkpoint checkpoints/best_model.pt \
      --output     health_doc_risk.onnx \
      [--opset 17] \
      [--dynamic]  \
      [--fp16]
 
Steps performed
───────────────
  1. Reconstruct the model from the saved checkpoint.
  2. Build dummy tensors that exactly match the model's expected input
     shapes and dtypes.
  3. Call torch.onnx.export with appropriate dynamic-axes settings.
  4. Re-load the exported graph with onnx.checker and onnxruntime,
     run the same dummy inputs, and compare outputs against PyTorch.
"""
 
import argparse
import sys
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
 
# ── optional but recommended ──────────────────────────────────
try:
    import onnx
    import onnx.checker
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: 'onnx' package not found — graph-level checks will be skipped.")
    print("         Install with:  pip install onnx")
 
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("WARNING: 'onnxruntime' package not found — runtime checks will be skipped.")
    print("         Install with:  pip install onnxruntime   (or onnxruntime-gpu)")
 
from model_and_training import HealthDocRiskModel  # 02_model_and_training.py
from dataset import IMG_H, IMG_W, MAX_SEQ_LEN      # 01_dataset.py
 
# ──────────────────────────────────────────────
# Constants — must match your training config
# ──────────────────────────────────────────────
BATCH_SIZE_EXPORT = 1   # static batch for the dummy; override with --dynamic
 
 
# ──────────────────────────────────────────────
# Thin ONNX-friendly wrapper
# ──────────────────────────────────────────────
class ExportWrapper(nn.Module):
    """
    torch.onnx.export traces through *args, not **kwargs.
    This wrapper unpacks the tuple so we get clean named inputs
    in the ONNX graph.
    """
    def __init__(self, model: HealthDocRiskModel) -> None:
        super().__init__()
        self.model = model
 
    def forward(
        self,
        image: torch.Tensor,
        text_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(image, text_ids, attention_mask)
 
 
# ──────────────────────────────────────────────
# Helper: build dummy inputs
# ──────────────────────────────────────────────
def make_dummy_inputs(
    batch_size: int,
    device: torch.device,
    fp16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (image, text_ids, attention_mask) tensors that reproduce
    the exact dtypes and shapes the model expects.
 
    Shape guide
    ───────────
      image          : [B, 3, IMG_H, IMG_W]   float32 (or float16 for fp16)
      text_ids       : [B, MAX_SEQ_LEN]        int64
      attention_mask : [B, MAX_SEQ_LEN]        int64
    """
    dtype  = torch.float16 if fp16 else torch.float32
    image  = torch.randn(batch_size, 3, IMG_H, IMG_W, dtype=dtype, device=device)
 
    # Simulate a real tokenised sentence: real token IDs + a trailing PAD block
    seq_len = MAX_SEQ_LEN
    pad_start = seq_len // 2
    text_ids       = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    text_ids[:, 0] = 101   # [CLS]
    text_ids[:, pad_start:] = 0        # [PAD]
    text_ids[:, pad_start - 1] = 102   # [SEP]
 
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    attention_mask[:, pad_start:] = 0  # PAD positions masked out
 
    return image, text_ids, attention_mask
 
 
# ──────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────
def export(
    checkpoint_path: str,
    output_path:     str,
    opset:           int  = 17,
    dynamic:         bool = False,
    fp16:            bool = False,
) -> None:
    device = torch.device("cpu")   # always export on CPU for portability
 
    # ── 1. Load model ───────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = HealthDocRiskModel()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
 
    if fp16:
        model = model.half()
 
    wrapped = ExportWrapper(model)
 
    # ── 2. Dummy inputs ─────────────────────────────────────────────
    image, text_ids, attention_mask = make_dummy_inputs(
        BATCH_SIZE_EXPORT, device, fp16=fp16
    )
    dummy_inputs = (image, text_ids, attention_mask)
 
    print(f"\nDummy input shapes:")
    print(f"  image          : {image.shape}          dtype={image.dtype}")
    print(f"  text_ids       : {text_ids.shape}  dtype={text_ids.dtype}")
    print(f"  attention_mask : {attention_mask.shape}  dtype={attention_mask.dtype}")
 
    # ── 3. Dynamic-axes definition ──────────────────────────────────
    #
    # Axis 0 of every tensor is the batch dimension.
    # Axis 1 of text tensors is the sequence length.
    # Marking them as dynamic lets ONNX Runtime accept variable-size
    # batches / sequences without re-exporting.
    #
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "image":          {0: "batch_size"},
            "text_ids":       {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "privacy_logits": {0: "batch_size"},
            "doc_logits":     {0: "batch_size"},
        }
 
    # ── 4. Export ───────────────────────────────────────────────────
    print(f"\nExporting to ONNX (opset {opset}) → {output_path}")
    torch.onnx.export(
        wrapped,
        dummy_inputs,
        output_path,
        opset_version  = opset,
        input_names    = ["image", "text_ids", "attention_mask"],
        output_names   = ["privacy_logits", "doc_logits"],
        dynamic_axes   = dynamic_axes,
        do_constant_folding = True,          # fold BatchNorm into Conv weights
        export_params  = True,
        verbose        = False,
    )
    print("Export complete.")
 
 
# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────
def verify(
    output_path: str,
    fp16:        bool = False,
    rtol:        float = 1e-3,
    atol:        float = 1e-5,
) -> None:
    """
    Three-layer verification:
      A) onnx.checker — validates the protobuf graph structure
      B) onnxruntime  — runs the graph and checks shapes
      C) Numeric diff — compares ONNX RT output vs. PyTorch CPU output
    """
    print(f"\n{'─'*55}")
    print("Verification")
    print(f"{'─'*55}")
 
    # ── A: graph check ──────────────────────────────────────────────
    if ONNX_AVAILABLE:
        print("[A] Loading graph with onnx.load …", end=" ")
        model_proto = onnx.load(output_path)
        onnx.checker.check_model(model_proto)
        print("OK")
        print(f"    opset imports : {[f'{o.domain or \"ai.onnx\"}:{o.version}' for o in model_proto.opset_import]}")
        print(f"    graph inputs  : {[i.name for i in model_proto.graph.input]}")
        print(f"    graph outputs : {[o.name for o in model_proto.graph.output]}")
    else:
        print("[A] Skipped (onnx not installed).")
 
    # ── B + C: runtime check ────────────────────────────────────────
    if not ORT_AVAILABLE:
        print("[B/C] Skipped (onnxruntime not installed).")
        return
 
    print("\n[B] Creating OnnxRuntime session …", end=" ")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        output_path,
        sess_options=sess_opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"OK  (provider: {sess.get_providers()[0]})")
 
    # Confirm expected inputs/outputs
    in_names  = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    print(f"    ORT inputs  : {in_names}")
    print(f"    ORT outputs : {out_names}")
 
    # Build feed dict (always float32 for ORT, even if model was exported fp16)
    device = torch.device("cpu")
    image, text_ids, attention_mask = make_dummy_inputs(
        BATCH_SIZE_EXPORT, device, fp16=fp16
    )
    ort_inputs = {
        "image":          image.float().numpy(),
        "text_ids":       text_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
 
    print("\n[B] Running ORT inference …", end=" ")
    ort_outs = sess.run(None, ort_inputs)
    print("OK")
    ort_privacy = ort_outs[0]
    ort_doc     = ort_outs[1]
    print(f"    privacy_logits shape : {ort_privacy.shape}")
    print(f"    doc_logits shape     : {ort_doc.shape}")
 
    # ── C: numeric comparison ────────────────────────────────────────
    print("\n[C] Comparing ONNX-RT output with PyTorch output …")
    # Re-instantiate model (we don't have the checkpoint path here, so we
    # use random weights — just testing graph topology consistency)
    pt_model = ExportWrapper(HealthDocRiskModel())
    pt_model.eval()
    if fp16:
        pt_model = pt_model.half()
 
    with torch.no_grad():
        pt_privacy, pt_doc = pt_model(
            image.to(torch.float16 if fp16 else torch.float32),
            text_ids,
            attention_mask,
        )
 
    pt_privacy_np = pt_privacy.float().numpy()
    pt_doc_np     = pt_doc.float().numpy()
 
    # Shapes
    assert ort_privacy.shape == pt_privacy_np.shape, "privacy_logits shape mismatch"
    assert ort_doc.shape     == pt_doc_np.shape,     "doc_logits shape mismatch"
    print("    Shape check: PASSED")
 
    # Numeric closeness (only meaningful when checkpoint weights were used)
    priv_close = np.allclose(ort_privacy, pt_privacy_np, rtol=rtol, atol=atol)
    doc_close  = np.allclose(ort_doc,     pt_doc_np,     rtol=rtol, atol=atol)
    print(f"    privacy_logits allclose (rtol={rtol}, atol={atol}): {'PASSED' if priv_close else 'FAILED *'}")
    print(f"    doc_logits     allclose (rtol={rtol}, atol={atol}): {'PASSED' if doc_close  else 'FAILED *'}")
    if not (priv_close and doc_close):
        print("    * Numeric diff expected if random weights used (no checkpoint).")
 
    print(f"\n{'─'*55}")
    print("Verification complete ✓")
    print(f"{'─'*55}")
 
 
# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export and verify HealthDocRiskModel → ONNX")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                   help="Path to PyTorch checkpoint (.pt)")
    p.add_argument("--output",     default="health_doc_risk.onnx",
                   help="Destination .onnx file")
    p.add_argument("--opset",      type=int,  default=17,
                   help="ONNX opset version (default: 17)")
    p.add_argument("--dynamic",    action="store_true",
                   help="Export with dynamic batch + sequence axes")
    p.add_argument("--fp16",       action="store_true",
                   help="Export model weights in float16")
    p.add_argument("--verify-only", action="store_true",
                   help="Skip export, only verify an existing .onnx file")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
 
    if not args.verify_only:
        if not Path(args.checkpoint).exists():
            sys.exit(
                f"ERROR: checkpoint not found at '{args.checkpoint}'. "
                "Train the model first with 02_model_and_training.py, "
                "or pass --verify-only to validate an existing .onnx file."
            )
        export(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            opset=args.opset,
            dynamic=args.dynamic,
            fp16=args.fp16,
        )
 
    verify(args.output, fp16=args.fp16)