import torch

def train_one_epoch(clip_model, arcface_head, loader, optimizer, device, epoch):
    clip_model.train()
    arcface_head.train()
    total_loss = 0.0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # BEFORE forward pass

        embeddings = clip_model.encode_image(imgs).float()
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        if torch.isnan(embeddings).any():
            print(f'  WARNING: NaN embeddings at batch {batch_idx}, skipping.')
            continue

        loss = arcface_head(embeddings, labels)

        if torch.isnan(loss):
            print(f'  WARNING: NaN loss at batch {batch_idx}, skipping.')
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(clip_model.parameters()) + list(arcface_head.parameters()), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(f'  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}')

    return total_loss / len(loader)