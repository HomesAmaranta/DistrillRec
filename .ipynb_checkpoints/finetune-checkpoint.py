import fire
import torch
from model import RecSys
from dataset import SeqDataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from torch import nn
from tqdm.auto import tqdm
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def collate_fn(batch_samples):
    seqs, labels = zip(*batch_samples)
    max_len = max(max([len(seq) for seq in seqs]), 2)
    inputs = [[0] * (max_len - len(seq)) + seq for seq in seqs]
    inputs_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]
    labels = [[label] for label in labels]
    inputs, inputs_mask, labels = torch.LongTensor(
        inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

    return inputs, inputs_mask, labels


def collate_fn_val(batch_samples):
    seqs, labels = zip(*batch_samples)
    max_len = max(max([len(seq) for seq in seqs]), 2)
    inputs = [[0] * (max_len - len(seq)) + seq for seq in seqs]
    inputs_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]
    labels = [[label] for label in labels]
    inputs, inputs_mask, labels = torch.LongTensor(
        inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

    return inputs, inputs_mask, labels


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch-1)*len(dataloader)

    model.train()
    for step, (inputs, inputs_mask, labels) in enumerate(dataloader, start=1):
        inputs, inputs_mask,labels = inputs.to(device), inputs_mask.to(device), labels.to(device)
        pred = model(inputs, inputs_mask)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(
            f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for inputs, inputs_mask, labels in dataloader:
            inputs, inputs_mask = inputs.to(
                device), inputs_mask.to(device), labels.to(device)
            pred = model(inputs, inputs_mask)
            correct += (pred.argmax(1) ==
                        labels).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct


def run(base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        data: str = "Beauty",
        warmup_steps: int = 100,
        lr_scheduler: str = "cosine",
        epoch_num: int = 1,
        learning_rate: float = 3e-4,
        maxlen: int = 200,
        item_embed_hidden_units: int = 50):
    dataset = SeqDataset("./"+data+"_processed.txt", maxlen)
    item_embed=pickle.load(open("./"+data+"_SASRec_item_embed.pkl", 'rb'))
    model = RecSys(output_dim=dataset.item_max,
                   input_dim=item_embed_hidden_units,
                   base_model=base_model,
                   item_embed=item_embed)
    train_dataloader = DataLoader(
        dataset.train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(
        dataset.val_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epoch_num*len(train_dataloader),
    )
    total_loss=0
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(
            train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
        test_loop(valid_dataloader, model, mode='Valid')
    print("Done!")
    torch.save(model.state_dict(), "model.bin")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(run)
