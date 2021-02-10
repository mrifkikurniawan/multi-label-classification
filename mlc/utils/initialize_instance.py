
__all__ = ["initialize_dataset", "initialize_loss"]

def initialize_loss(module: object, loss: str, **args):
    loss_name = loss
    print(f"initializing loss function: {loss_name}")
    loss_ = getattr(module, loss_name)
    loss_ = loss_(**args)
    return loss_

def initialize_dataset(module: object, dataset: str, **args):
    print(f"initializing dataset {dataset}")

    dataset_ = getattr(module, dataset)
    dataset_ = dataset_(**args)
    return dataset_