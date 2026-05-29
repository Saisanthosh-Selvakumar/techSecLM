
"""
01_dataset.py
Health Document Risk Dataset — preprocessing + PyTorch Dataset class.
 
Inputs expected per sample
───────────────────────────
  • A health document image  (JPEG / PNG / TIFF …)
  • A companion CSV with columns:
        document_id  | image_path | text | privacy_risk_label | doc_risk_label
    where *_label columns are integer class indices (0 = low, 1 = medium, 2 = high)
 
Outputs from __getitem__
────────────────────────
  {
    "image"          : FloatTensor [3, IMG_H, IMG_W],
    "text_ids"       : LongTensor  [MAX_SEQ_LEN],
    "attention_mask" : LongTensor  [MAX_SEQ_LEN],
    "privacy_label"  : LongTensor  [],
    "doc_label"      : LongTensor  [],
  }
"""
 
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
 
# ──────────────────────────────────────────────
# Global constants — change these to suit your setup
# ──────────────────────────────────────────────
IMG_H, IMG_W   = 224, 224          # spatial size fed to CNN backbone
MAX_SEQ_LEN    = 128               # token budget for text encoder
TOKENIZER_NAME = "bert-base-uncased"
IMG_MEAN       = [0.485, 0.456, 0.406]   # ImageNet stats (fine for transfer)
IMG_STD        = [0.229, 0.224, 0.225]
 
 
# ──────────────────────────────────────────────
# Image helpers (OpenCV)
# ──────────────────────────────────────────────
def load_and_preprocess_image(path: str) -> np.ndarray:
    """
    Read an image with OpenCV, resize to (IMG_H × IMG_W),
    convert BGR → RGB, normalise to [0, 1], then standardise
    with ImageNet mean/std.
 
    Returns: float32 ndarray of shape (3, IMG_H, IMG_W)
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        # Return a blank image rather than crashing the DataLoader worker
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    img = img.astype(np.float32) / 255.0                      # → [0, 1]
 
    # Per-channel standardisation  (broadcast over H × W)
    mean = np.array(IMG_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(IMG_STD,  dtype=np.float32).reshape(1, 1, 3)
    img  = (img - mean) / (std + 1e-6)
 
    return img.transpose(2, 0, 1)                             # HWC → CHW
 
 
def enhance_document_image(path: str) -> np.ndarray:
    """
    Optional enhancement pipeline for scanned / low-quality docs:
      • CLAHE contrast enhancement on the L channel (LAB colour space)
      • Gaussian denoising
      • Binarisation via Otsu thresholding (kept as a greyscale mask
        but NOT used directly — returned for debug / OCR purposes)
 
    Returns the enhanced colour image as float32 CHW ndarray.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((3, IMG_H, IMG_W), dtype=np.float32)
 
    # CLAHE on luminance
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    img   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
 
    # Gaussian denoising
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMG_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(IMG_STD,  dtype=np.float32).reshape(1, 1, 3)
    img  = (img - mean) / (std + 1e-6)
    return img.transpose(2, 0, 1)
 
 
# ──────────────────────────────────────────────
# Text helpers (pandas + HuggingFace tokeniser)
# ──────────────────────────────────────────────
def clean_text(series: pd.Series) -> pd.Series:
    """
    Vectorised pandas cleaning:
      • lower-case
      • strip whitespace
      • remove non-ASCII characters common in OCR artefacts
      • collapse multiple spaces
    """
    return (
        series.str.lower()
              .str.strip()
              .str.encode("ascii", errors="ignore").str.decode("ascii")
              .str.replace(r"\s+", " ", regex=True)
    )
 
 
# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class HealthDocRiskDataset(Dataset):
    """
    Parameters
    ----------
    csv_path    : path to the metadata CSV described at the top of this file
    tokenizer   : a HuggingFace tokenizer instance (shared across workers)
    enhance     : if True, apply the CLAHE enhancement pipeline to images
    """
 
    def __init__(
        self,
        csv_path: str,
        tokenizer: AutoTokenizer,
        enhance: bool = False,
    ) -> None:
        self.df        = self._load_and_validate(csv_path)
        self.tokenizer = tokenizer
        self.enhance   = enhance
 
    # ── internal helpers ────────────────────────
    @staticmethod
    def _load_and_validate(csv_path: str) -> pd.DataFrame:
        required = {"document_id", "image_path", "text",
                    "privacy_risk_label", "doc_risk_label"}
        df = pd.read_csv(csv_path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
 
        # Clean text in-place
        df["text"] = clean_text(df["text"].fillna(""))
 
        # Coerce labels to int; drop rows with invalid labels
        for col in ("privacy_risk_label", "doc_risk_label"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["privacy_risk_label", "doc_risk_label"])
        df[["privacy_risk_label", "doc_risk_label"]] = (
            df[["privacy_risk_label", "doc_risk_label"]].astype(int)
        )
 
        df = df.reset_index(drop=True)
        print(f"[Dataset] Loaded {len(df)} samples from {csv_path}")
        return df
 
    # ── DataLoader interface ────────────────────
    def __len__(self) -> int:
        return len(self.df)
 
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
 
        # ---- image ----
        img_fn = load_and_preprocess_image if not self.enhance else enhance_document_image
        image  = torch.tensor(img_fn(row["image_path"]), dtype=torch.float32)
 
        # ---- text ----
        encoding = self.tokenizer(
            row["text"],
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_ids      = encoding["input_ids"].squeeze(0)       # [MAX_SEQ_LEN]
        attention_mask = encoding["attention_mask"].squeeze(0) # [MAX_SEQ_LEN]
 
        # ---- labels ----
        privacy_label = torch.tensor(row["privacy_risk_label"], dtype=torch.long)
        doc_label     = torch.tensor(row["doc_risk_label"],     dtype=torch.long)
 
        return {
            "image":          image,
            "text_ids":       text_ids,
            "attention_mask": attention_mask,
            "privacy_label":  privacy_label,
            "doc_label":      doc_label,
        }
 
 
# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import os, tempfile
 
    # ---- build a tiny synthetic CSV ----
    rows = []
    for i in range(6):
        rows.append({
            "document_id":     f"doc_{i:03d}",
            "image_path":      "nonexistent.jpg",   # will produce blank img
            "text":            f"Patient ID 123  SSN 456-78-9012  diagnosis: condition {i}",
            "privacy_risk_label": i % 3,
            "doc_risk_label":     (i + 1) % 3,
        })
 
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pd.DataFrame(rows).to_csv(f, index=False)
        tmp_csv = f.name
 
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    ds        = HealthDocRiskDataset(tmp_csv, tokenizer)
    loader    = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
 
    batch = next(iter(loader))
    print("\n── Batch shapes ──")
    for k, v in batch.items():
        print(f"  {k:20s}: {tuple(v.shape)}  dtype={v.dtype}")
 
    os.unlink(tmp_csv)
    print("\n[01_dataset.py] smoke-test passed ✓")