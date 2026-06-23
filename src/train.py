import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(clip_model, arcface_head, loader, optimizer, device, epoch):
    clip_model.train()
    arcface_head.train()
    total_loss = 0.0
    batch_losses = []
    # mixed precision training to save memory, scales loss
    scaler = GradScaler()        # TODO move outside the training loop

    for batch_idx, (imgs, labels) in enumerate(loader):
        # move everything to gpu if posible
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass in float16 where possible
        with autocast():
            # extract and L2-normalise embeddings
            embeddings = clip_model.encode_image(imgs).float()
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            # NaN check: skip batch if invalid values
            if torch.isnan(embeddings).any():
                print(f'  WARNING: NaN embeddings at batch {batch_idx}, skipping.')
                continue
            loss = arcface_head(embeddings, labels)
            # NaN check: skip batch if exploded loss
            if torch.isnan(loss):
                print(f'  WARNING: NaN loss at batch {batch_idx}, skipping.')
                continue
        # backward pass in float16 and then convert grad to float32
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # grad clipping: try to avoid exploding gradients during backprop
        torch.nn.utils.clip_grad_norm_(
            list(clip_model.parameters()) + list(arcface_head.parameters()), 1.0)    #scale all grad.s down if above 1.0
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        total_loss   += loss_val
        # append loss val to later plot
        batch_losses.append(loss_val)
        # print loss every 20 values
        if (batch_idx + 1) % 20 == 0:
            print(f'  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}')
    # average loss over all batches
    return total_loss / len(loader)
