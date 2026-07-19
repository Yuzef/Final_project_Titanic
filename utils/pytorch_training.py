import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.pytorch_models import TitanicMLPNet

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}.")
        return device
    
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False.")
    if device_name == "mps" and not torch.backends.mps.is_available():
           raise ValueError("MPS was requested, but torch.backends.mps.is_available() is False.")

    device = torch.device(device_name)
    print(f"Using device: {device}.")
    return device

def build_loss(loss_cfg):
    params = dict(loss_cfg.params)

    if loss_cfg.name == "cross_entropy":
        return nn.CrossEntropyLoss(**params)
    
    raise ValueError(f"Unknown loss: {loss_cfg.name}")

def build_optimizer(model, optimizer_cfg):
    params = dict(optimizer_cfg.params)

    if optimizer_cfg.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            **params,
        )
    
    if optimizer_cfg.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            **params,
        )
    
    raise ValueError(f"Unknown optimizer: {optimizer_cfg.name}")

# output_dim = 2 (бинарная классификация).
def train_pytorch_model(X_train, y_train, model_cfg, dl_cfg, seed):
    set_random_seed(seed) 
    device = resolve_device(dl_cfg.training.device)

    # numpy нужен как мост из pandas-таблиц в чистые числовые tensors.
    X_np = np.asarray(X_train, dtype=np.float32)
    y_np = np.asarray(y_train, dtype=np.int64)    

    # Кол-во входных признаков.
    input_dim = X_np.shape[1]
    output_dim = model_cfg.params.output_dim

    model = TitanicMLPNet(
        input_dim=input_dim,
        hidden_dim=model_cfg.params.hidden_dim,
        output_dim=output_dim,
        activation_name=model_cfg.params.activation,
    ).to(device)

    # TensorDataset связывает X и y попарно:
    dataset = TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.long),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=dl_cfg.dataloader_params.batch_size,
        shuffle=dl_cfg.dataloader_params.shuffle,
        drop_last=dl_cfg.dataloader_params.drop_last,
        num_workers=dl_cfg.dataloader_params.num_workers,
        pin_memory=dl_cfg.dataloader_params.pin_memory,
    )          
    
    loss_fn = build_loss(dl_cfg.loss)

    optimizer = build_optimizer(
        model=model,
        optimizer_cfg=dl_cfg.optimizer,
    )   

    model.train()

    for epoch in range(dl_cfg.training.num_epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            # Переносим датасет батчами внутри цикла для
            # оптимизированной работы с памятью - "batch-wise transfer".
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # В PyTorch градиенты по умолчанию накапливаются.
            # Поэтому перед новым batch нужно очистить старые градиенты.
            optimizer.zero_grad()

            # logits — это сырые score, не вероятности.
            logits = model(X_batch)
            # Для CrossEntropyLoss не нужно заранее делать softmax.
            # Она сама внутри умеет работать с logits.
            loss = loss_fn(logits, y_batch) 

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Печатаем среднюю ошибку за эпоху.
        if dl_cfg.training.verbose:
            print(f"Epoch {epoch + 1}: loss={epoch_loss / len(dataloader):.4f}")

    return model           


