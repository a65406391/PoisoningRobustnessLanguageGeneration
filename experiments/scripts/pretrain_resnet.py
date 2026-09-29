import torch
from tqdm import trange

from certifiable_learning_stability.inference import accuracy
from certifiable_learning_stability.models.resnet import Resnet18
from data_sets.cifar import CIFAR
from data_sets.dset_type import DsetType
from experiments.reproducibility import get_device, make_reproducible

make_reproducible(42)
device = get_device(index=0)

train_set = CIFAR(DsetType.TRAIN_FULL, 42, cifar_100=True)
test_set = CIFAR(DsetType.TEST, 42, cifar_100=True)

batch_size = 512
epochs = 30
lr = 0.001
weight_decay = 5e-4


model = Resnet18(output_dim=100).to(device)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()


# Train
batch_acc = lambda preds, labels: (preds == labels).float().mean().item()
compute_preds = lambda out: torch.argmax(out, dim=1).int()
progress_bar = trange(
    epochs,
    desc="Epoch",
)

for epoch in progress_bar:
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = compute_preds(outputs)
        acc = batch_acc(preds, labels.int())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({"Epoch": epoch + 1, "Batch": i + 1, "Loss": loss.item(), "Accuracy": acc})

accuracy_score = accuracy(model, test_loader, device)
print(f"Inference accuracy on whole clean test set after training: {accuracy_score:.4f}")

# Save the model
torch.save(model.state_dict(), "resnet18_cifar100_pretrained.pt")
