from torchvision import datasets, transforms


def get_mnist(dataset_path):
    path_str = str(dataset_path.resolve())
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, )),
    ])
    train_data = datasets.MNIST(path_str, train=True, download=True, transform=transform)
    train_eval_data = train_data
    test_data = datasets.MNIST(path_str, train=False, download=True, transform=transform)
    return train_data, train_eval_data, test_data


def get_cifar10(dataset_path, proper_normalization=True):
    if proper_normalization:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    path_str = str(dataset_path.resolve())
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_data = datasets.CIFAR10(path_str, train=True, download=True, transform=transform_train)
    train_eval_data = datasets.CIFAR10(path_str, train=True, download=True, transform=transform_eval)
    test_data = datasets.CIFAR10(path_str, train=False, download=True, transform=transform_eval)
    return train_data, train_eval_data, test_data
