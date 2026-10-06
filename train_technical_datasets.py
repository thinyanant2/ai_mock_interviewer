import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup # type: ignore

from config import (
    DEVICE,
    EPOCHS,
    FILE_PATH,
    LEARNING_RATE,
    MODEL_NAME,
    SAVE_MODEL_PATH,
    SEED,
    TOLERANCE_THRESHOLD,
    WEIGHT_DECAY,
    set_seed,
)
from dataset import create_dataloaders, load_and_preprocess_data
from model import MultiOutputRegressor


# ==========================================
# 1. EVALUATION FUNCTION
# ==========================================
def evaluate_performance(
    model, dataloader, device, tolerance=TOLERANCE_THRESHOLD
):
    """
    Validation / Test Dataset ပေါ်တွင် Model ၏ MSE Loss၊ Tolerance Accuracy (±0.10)၊
    R² Score နှင့် MAE Score များကို တွက်ချက်ပေးသော Function ဖြစ်ပါသည်။
    """
    model.eval()
    all_preds = []
    all_targets = []
    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Tolerance Accuracy တွက်ချက်ခြင်း (သတ်မှတ်ထားသော ± tolerance threshold အတွင်း ရောက်မရောက်)
    abs_errors = np.abs(all_preds - all_targets)
    within_tol = abs_errors <= tolerance
    overall_acc = np.mean(within_tol) * 100
    per_target_acc = np.mean(within_tol, axis=0) * 100

    r2_scores = r2_score(all_targets, all_preds, multioutput="raw_values")
    mae_scores = mean_absolute_error(
        all_targets, all_preds, multioutput="raw_values"
    )

    return (
        avg_loss,
        overall_acc,
        per_target_acc,
        r2_scores,
        mae_scores,
        all_preds,
        all_targets,
    )


# ==========================================
# 2. MAIN TRAINING LOOP FUNCTION
# ==========================================
def run_training():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # Data ဖတ်ယူခြင်းနှင့် DataLoaders ဖန်တီးခြင်း
    train_df, val_df, test_df = load_and_preprocess_data(FILE_PATH)
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        train_df, val_df, test_df
    )

    # Model Initialize ပြုလုပ်ခြင်း
    model = MultiOutputRegressor(MODEL_NAME).to(DEVICE)

    # Optimizer, Loss Function နှင့် Scheduler ပြင်ဆင်ခြင်း
    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        print(
            f"\n================ Epoch {epoch + 1}/{EPOCHS} ================"
        )

        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Step {step + 1}/{len(train_loader)} | Train Loss:"
                    f" {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        val_loss, overall_acc, per_target_acc, r2_scores, mae_scores, _, _ = (
            evaluate_performance(model, val_loader, DEVICE)
        )

        print(f"\n[Epoch {epoch + 1} Summary]")
        print(
            f"Train MSE Loss: {avg_train_loss:.4f} | Val MSE Loss:"
            f" {val_loss:.4f}"
        )
        print(
            "Overall Accuracy (within"
            f" ±{TOLERANCE_THRESHOLD}): {overall_acc:.2f}%"
        )
        print(
            f"  • Accuracy score   -> Acc: {per_target_acc[0]:.2f}% | R²:"
            f" {r2_scores[0]:.4f} | MAE: {mae_scores[0]:.4f}"
        )
        print(
            f"  • Sentiment score  -> Acc: {per_target_acc[1]:.2f}% | R²:"
            f" {r2_scores[1]:.4f} | MAE: {mae_scores[1]:.4f}"
        )
        print(
            f"  • Anxiety score    -> Acc: {per_target_acc[2]:.2f}% | R²:"
            f" {r2_scores[2]:.4f} | MAE: {mae_scores[2]:.4f}"
        )

        # Best Validation Loss ရလျှင် Model Checkpoint သိမ်းဆည်းခြင်း
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"--> Saved new best checkpoint to {SAVE_MODEL_PATH}")

    # ==========================================
    # 3. FINAL TEST SET EVALUATION
    # ==========================================
    print("\nLoading best model for Final Test Evaluation...")
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))

    test_loss, test_acc, test_target_acc, test_r2, test_mae, _, _ = (
        evaluate_performance(model, test_loader, DEVICE)
    )

    print("\n" + "=" * 60)
    print("                   FINAL TEST SET RESULTS                   ")
    print("=" * 60)
    print(f"Test MSE Loss: {test_loss:.4f}")
    print(
        "Overall Test Accuracy (within"
        f" ±{TOLERANCE_THRESHOLD}): {test_acc:.2f}%\n"
    )
    print(
        f"1. target_accuracy  -> Acc: {test_target_acc[0]:.2f}% | R²:"
        f" {test_r2[0]:.4f} | MAE: {test_mae[0]:.4f}"
    )
    print(
        f"2. target_sentiment -> Acc: {test_target_acc[1]:.2f}% | R²:"
        f" {test_r2[1]:.4f} | MAE: {test_mae[1]:.4f}"
    )
    print(
        f"3. target_anxiety   -> Acc: {test_target_acc[2]:.2f}% | R²:"
        f" {test_r2[2]:.4f} | MAE: {test_mae[2]:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    run_training()