import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.pytorch_models import TitanicMLPNet
from sklearn.metrics import accuracy_score

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

def evaluate_pytorch_model(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            predictions = torch.argmax(logits, dim=1)

            batch_size = y_batch.size(0)
            # loss.item() - это средний loss на текущем batch.
            total_loss += loss.item() * batch_size
            correct_predictions += (predictions == y_batch).sum().item()
            total_examples += batch_size
    
    mean_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    return mean_loss, accuracy


def train_pytorch_model(
    X_train,
    y_train,
    model_cfg,
    dl_cfg,
    seed,
    X_valid=None,
    y_valid=None,
):
    set_random_seed(seed) 
    device = resolve_device(dl_cfg.training.device)

    # numpy нужен как мост из pandas-таблиц в чистые числовые tensors.
    X_np = np.asarray(X_train, dtype=np.float32)
    y_np = np.asarray(y_train, dtype=np.int64)    

    # Кол-во входных признаков.
    input_dim = X_np.shape[1]
    output_dim = model_cfg.params.output_dim

    if model_cfg.params.architecture != "mlp":
        raise ValueError(
            f"Unknown PyTorch architecture: {model_cfg.params.architecture}"
        )

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
        persistent_workers=(
            dl_cfg.dataloader_params.persistent_workers
            and dl_cfg.dataloader_params.num_workers > 0
        )
    )

    valid_dataloader = None

    if X_valid is not None and y_valid is not None:
        X_valid_np = np.asarray(X_valid, dtype=np.float32)
        y_valid_np = np.asaraay(y_valid, dtype=np.int64)

        valid_dataset = TensorDataset(
            torch.tensor(X_valid_np, dtype=torch.float32),
            torch.tensor(y_valid_np, dtype=torch.long)
        )          

        valid_dataloader = TensorDataset(
            torch.tensor(X_valid_np, dtype=torch.float32)
            torch.tensor(y_valid_np, dtype=torch.long),
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=dl_cfg.dataloader_params.batch_size,
            # Для validation всегда shuffle=False,
            # потому что мы не обучаемся, а просто оцениваем.
            shuffle=False,
            drop_last=False,
            num_workers=dl_cfg.dataloader_params.num_workers,
            pin_memory=dl_cfg.dataloader_params.pin_memory,
            persistent_workers=(
                dl_cfg.dataloader_params.persistent_workers
                and dl_cfg.dataloader_params.num_workers > 0
            ),
        )
    
    loss_fn = build_loss(dl_cfg.loss)

    optimizer = build_optimizer(
        model=model,
        optimizer_cfg=dl_cfg.optimizer,
    )   

    history = []

    for epoch in range(dl_cfg.training.num_epochs):
        # evaluate_pytorch_model меняет режим в конце эпохи 
        # на eval во время подсчёта метрик, если выполняется условие,
        # поэтому модель надо возвращать обратно в train режим.
        model.train()

        epoch_loss = 0.0
        correct_predictions = 0.0
        total_examples = 0

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

            # Для вывода на график correct_predictions
            predictions = torch.argmax(logits, dim=1) 

            loss.backward()
            optimizer.step()

            batch_size = y_batch.size(0)
            epoch_loss += loss.item() * batch_size
            correct_predictions += (predictions == y_batch).sum().item()
            total_examples += batch_size

        train_loss = epoch_loss / total_examples
        train_accuracy = correct_predictions / total_examples

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }

        should_evaluate = (
            valid_dataloader is not None
            and (epoch + 1) % dl_cfg.training.eval_every_n_epochs == 0 
        )

        if should_evaluate:
            valid_loss, valid_accuracy = evaluate_pytorch_model(
                model=model,
                dataloader=valid_dataloader,
                loss_fn=loss_fn,
                device=device,
            )

            row["valid_loss"] = valid_loss
            row["valid_accuracy"] = valid_accuracy
        
        history.append(row)

        if dl_cfg.training.verbose:
            message = (
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"train_accuracy={train_accuracy:.4f}"
            )

            if "valid_loss" in row:
                message += (
                    f", valid_loss={row['valid_loss']:.4f}, "
                    f"valid_accuracy={row['valid_accuracy']:.4f}"
                )
            
            print(message)

    return model, history           

def predict_pytorch_model(model, X, dl_cfg):
    device = resolve_device(dl_cfg.training.device)

    X_np = np.asarray(X, dtype=np.float32)

    X_tensor = torch.tensor(
        X_np,
        dtype=torch.float32,
    ).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        # dim=1 означает: ищем максимум по классам внутри каждой строки.
        # dim=0 -> идти по строкам сверху вниз
        # dim=1 -> идти по колонкам внутри строки
        predictions = torch.argmax(logits, dim=1)
    
    # Возращаем в виде numpy результата, т.к. далее используется 
    # sklearn ф-ция accuracy_score .
    return predictions.cpu().numpy()
