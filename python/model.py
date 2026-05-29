"""
02_model_and_training.py
Multi-modal health-document risk classifier.
 
Architecture overview
─────────────────────
  ┌──────────────┐        ┌───────────────┐
  │  CNN Image   │        │  BERT Text    │
  │  Backbone    │        │  Encoder      │
  │ (ResNet-18)  │        │ (bert-base)   │
  └──────┬───────┘        └──────┬────────┘
         │  img_feat [B,512]     │ txt_feat [B,768]
         └──────────┬────────────┘
                    │  concat → [B, 1280]
             ┌──────▼──────┐
             │  Fusion MLP │
             │  1280→512→256
             └──────┬──────┘
           ┌────────┴────────┐
    ┌──────▼──────┐   ┌──────▼──────┐
    │  Privacy    │   │  Document   │
    │  Head       │   │  Risk Head  │
    │  256→3      │   │  256→3      │
    └─────────────┘   └─────────────┘
 
Both heads output raw logits for 3-class cross-entropy.
Total loss = privacy_loss + doc_loss  (equal weighting).
"""
 
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel, AutoTokenizer
 
from dataset import (                  # re-use 01_dataset.py
    HealthDocRiskDataset,
    TOKENIZER_NAME,
    IMG_H, IMG_W, MAX_SEQ_LEN,
)
 
# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
NUM_CLASSES  = 3       # low / medium / high risk
IMG_FEAT_DIM = 512     # ResNet-18 penultimate layer
TXT_FEAT_DIM = 768     # bert-base hidden size
FUSED_DIM    = 256
DROPOUT_P    = 0.3
LR           = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 20
BATCH_SIZE   = 16
GRAD_CLIP    = 1.0     # max gradient norm
 
 
# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class HealthDocRiskModel(nn.Module):
    """
    Multi-modal encoder → dual classification heads.
 
    Parameters
    ----------
    num_classes   : number of risk levels (default 3)
    freeze_bert   : if True, only fine-tune the fusion MLP + heads
    freeze_cnn    : if True, only fine-tune from layer4 + heads
    """
 
    def __init__(
        self,
        num_classes:  int  = NUM_CLASSES,
        freeze_bert:  bool = False,
        freeze_cnn:   bool = False,
    ) -> None:
        super().__init__()
 
        # ── Image encoder ──────────────────────────────────────────────
        cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Drop the final FC layer; keep average-pool → 512-d vector
        self.image_encoder = nn.Sequential(*list(cnn.children())[:-1])
        self.img_proj = nn.Sequential(
            nn.Flatten(),                       # [B, 512, 1, 1] → [B, 512]
            nn.LayerNorm(IMG_FEAT_DIM),
        )
 
        if freeze_cnn:
            # Only fine-tune layer4 and beyond
            for name, param in self.image_encoder.named_parameters():
                if "layer4" not in name:
                    param.requires_grad = False
 
        # ── Text encoder ───────────────────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(TOKENIZER_NAME)
        # Use the [CLS] token → 768-d
        self.txt_proj = nn.Sequential(
            nn.Linear(TXT_FEAT_DIM, TXT_FEAT_DIM),
            nn.LayerNorm(TXT_FEAT_DIM),
            nn.GELU(),
        )
 
        if freeze_bert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
 
        # ── Cross-modal fusion ─────────────────────────────────────────
        fused_in = IMG_FEAT_DIM + TXT_FEAT_DIM          # 512 + 768 = 1280
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(512, FUSED_DIM),
            nn.LayerNorm(FUSED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
        )
 
        # ── Task heads ─────────────────────────────────────────────────
        self.privacy_head = nn.Linear(FUSED_DIM, num_classes)
        self.doc_head     = nn.Linear(FUSED_DIM, num_classes)
 
        self._init_heads()
 
    def _init_heads(self) -> None:
        for head in (self.privacy_head, self.doc_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
 
    # ── Forward pass ───────────────────────────────────────────────────
    def forward(
        self,
        image:          torch.Tensor,   # [B, 3, H, W]
        text_ids:       torch.Tensor,   # [B, seq_len]
        attention_mask: torch.Tensor,   # [B, seq_len]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        privacy_logits : [B, num_classes]
        doc_logits     : [B, num_classes]
        """
        # Image branch
        img_feat = self.image_encoder(image)      # [B, 512, 1, 1]
        img_feat = self.img_proj(img_feat)         # [B, 512]
 
        # Text branch
        txt_out  = self.text_encoder(
            input_ids=text_ids,
            attention_mask=attention_mask,
        )
        cls_tok  = txt_out.last_hidden_state[:, 0, :]  # [B, 768]
        txt_feat = self.txt_proj(cls_tok)
 
        # Fusion
        fused          = torch.cat([img_feat, txt_feat], dim=-1)  # [B, 1280]
        fused          = self.fusion(fused)                        # [B, 256]
 
        privacy_logits = self.privacy_head(fused)
        doc_logits     = self.doc_head(fused)
 
        return privacy_logits, doc_logits
 
 
# ──────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()
 
 
def train_one_epoch(
    model:     HealthDocRiskModel,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: Adam,
    device:    torch.device,
    scaler:    torch.cuda.amp.GradScaler | None,
) -> dict:
    model.train()
    total_loss  = privacy_acc_sum = doc_acc_sum = 0.0
 
    for batch in loader:
        image          = batch["image"].to(device)
        text_ids       = batch["text_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        privacy_label  = batch["privacy_label"].to(device)
        doc_label      = batch["doc_label"].to(device)
 
        optimizer.zero_grad(set_to_none=True)
 
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            p_logits, d_logits = model(image, text_ids, attention_mask)
            loss = criterion(p_logits, privacy_label) + criterion(d_logits, doc_label)
 
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
 
        total_loss     += loss.item()
        privacy_acc_sum += compute_accuracy(p_logits, privacy_label)
        doc_acc_sum     += compute_accuracy(d_logits, doc_label)
 
    n = len(loader)
    return {
        "loss":        total_loss / n,
        "privacy_acc": privacy_acc_sum / n,
        "doc_acc":     doc_acc_sum / n,
    }
 
 
@torch.no_grad()
def evaluate(
    model:     HealthDocRiskModel,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
    device:    torch.device,
) -> dict:
    model.eval()
    total_loss = privacy_acc_sum = doc_acc_sum = 0.0
 
    for batch in loader:
        image          = batch["image"].to(device)
        text_ids       = batch["text_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        privacy_label  = batch["privacy_label"].to(device)
        doc_label      = batch["doc_label"].to(device)
 
        p_logits, d_logits = model(image, text_ids, attention_mask)
        loss = criterion(p_logits, privacy_label) + criterion(d_logits, doc_label)
 
        total_loss      += loss.item()
        privacy_acc_sum += compute_accuracy(p_logits, privacy_label)
        doc_acc_sum     += compute_accuracy(d_logits, doc_label)
 
    n = len(loader)
    return {
        "loss":        total_loss / n,
        "privacy_acc": privacy_acc_sum / n,
        "doc_acc":     doc_acc_sum / n,
    }
 
 
# ──────────────────────────────────────────────
# Main training script
# ──────────────────────────────────────────────
def train(csv_path: str, output_dir: str = ".") -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
 
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
 
    # ---- Dataset & splits ----
    full_ds   = HealthDocRiskDataset(csv_path, tokenizer)
    val_size  = max(1, int(0.15 * len(full_ds)))
    test_size = max(1, int(0.10 * len(full_ds)))
    train_size = len(full_ds) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(
        full_ds, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
 
    kwargs = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)
 
    # ---- Model ----
    model     = HealthDocRiskModel(freeze_bert=True, freeze_cnn=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
 
    # Two-phase LR: backbone params get 10× smaller LR
    backbone_params = (
        list(model.image_encoder.parameters()) +
        list(model.text_encoder.parameters())
    )
    head_params = (
        list(model.img_proj.parameters()) +
        list(model.txt_proj.parameters()) +
        list(model.fusion.parameters())   +
        list(model.privacy_head.parameters()) +
        list(model.doc_head.parameters())
    )
    optimizer = Adam(
        [
            {"params": backbone_params, "lr": LR / 10},
            {"params": head_params,     "lr": LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
 
    best_val_loss = float("inf")
    best_ckpt     = os.path.join(output_dir, "best_model.pt")
 
    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
 
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val priv_acc {val_metrics['privacy_acc']:.3f} | "
            f"val doc_acc  {val_metrics['doc_acc']:.3f}"
        )
 
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "opt_state":   optimizer.state_dict(),
                    "val_loss":    best_val_loss,
                },
                best_ckpt,
            )
            print(f"  ✓ New best model saved → {best_ckpt}")
 
    # ---- Final test evaluation ----
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"\nTest results — loss: {test_metrics['loss']:.4f} | "
        f"priv_acc: {test_metrics['privacy_acc']:.3f} | "
        f"doc_acc: {test_metrics['doc_acc']:.3f}"
    )
 
 
if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "health_docs.csv"
    train(csv, output_dir="checkpoints")

