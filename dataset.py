from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def load_data(download:bool = True, root:str = './data', batch_size:int = 32):
    assert batch_size > 0 , 'batch_size must be larger than 0.'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    train_ds = datasets.MNIST(root = root, train = True, download = download, transform = transform)
    test_ds = datasets.MNIST(root = root, train = False, download = download, transform = transform)

    train_loader = DataLoader(dataset = train_ds, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_ds, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader
