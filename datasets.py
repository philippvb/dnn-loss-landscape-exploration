import torchvision
from torchvision import transforms

# copied by SlS but shortened


def getDataSet(name, data_directory, train_flag):
    dataset = None
    if name == "mnist":
        dataset = torchvision.datasets.MNIST(data_directory, train=train_flag,
                               download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ]))

    if name == "cifar10":
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=data_directory,
            train=train_flag,
            download=True,
            transform=transform_function)

    if name == "cifar100":
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR100(
            root=data_directory,
            train=train_flag,
            download=False,
            transform=transform_function)


    return dataset