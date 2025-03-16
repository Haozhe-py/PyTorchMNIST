import model as m
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = m.MNISTmodel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters())

train_loader, test_loader = dataset.load_data()


def train_step(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        continue
    return 0

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def fit(epochs:int = 20, patience:int = 3) -> None:
    assert epochs > 0 , 'epochs must be larger than 0.'
    assert patience >= 0, 'patience must be 0 or larger than 0.'
    max_acc = 0
    waiting = 0
    for i in range(1,epochs+1):
        model.train()
        train_step(i)
        print('\n\n')
        acc = test()
        if acc > max_acc:
            max_acc = acc
            waiting = 0
            torch.save(model, 'model.pt')                                                       #model checkpoint: save best only
        else:
            waiting += 1
        if waiting > patience:
            print(f'Early Stopping: at Epoch {i}/{epochs} . ')                                  #early stopping
            break
        continue
    return None

if __name__ == '__main__':
    fit()
    exit()