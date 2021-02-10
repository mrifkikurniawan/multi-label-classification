from pytorch_lightning import Trainer, seed_everything
from project.multi_label_classifier import MultiLabelClassifier
from mlc.utils import initialize_dataset
from mlc import datasets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

def test_multi_label_classifier():
    seed_everything(1234)

    train_transform = transforms.Compose([
                       transforms.ToPILImage(mode='RGBA'),
                       transforms.Resize((224, 224)), 
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomRotation(10),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])
    val_transform = transforms.Compose([
                     transforms.ToPILImage(mode='RGBA'),
                     transforms.Resize((224, 224)), 
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([
                      transforms.ToPILImage(mode='RGBA'),
                      transforms.Resize((224, 224)), 
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

    # dataset
    dataset_name = 'HPASingleCellClassification'
    root = '/media/nodeflux/DATA/my_repo/dataset/hpa-classificataion/train'
    metadata = '/media/nodeflux/DATA/my_repo/dataset/hpa-classificataion/hpa_check.csv'
    num_class = 100
    batch_size = 2

    dataset_train = initialize_dataset(datasets, dataset_name, root=root, metadata=metadata, num_class=num_class, transform=train_transform)
    dataset_test =  initialize_dataset(datasets, dataset_name, root=root, metadata=metadata, num_class=num_class, transform=test_transform)
    len_dataset = len(dataset_train)
    len_train = int(0.8*len_dataset)
    len_val = len_dataset - len_train
    dataset_train, dataset_val = random_split(dataset_train, [len_train, len_val])

    # dataloader
    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)
    test_loader = DataLoader(dataset_test, batch_size=1)

    model = 'efficientnet_b0' 
    learning_rate = 0.0001
    loss_functions = dict(module='binary_cross_entropy', weight=1)
    checkpoint_path = ''
    pretrained = True
    in_chans = 4

    model = MultiLabelClassifier(model, num_class, in_chans, learning_rate, loss_functions, checkpoint_path, pretrained)
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2, gpus=1)
    trainer.fit(model, train_loader, val_loader)

    results = trainer.test(test_dataloaders=test_loader)