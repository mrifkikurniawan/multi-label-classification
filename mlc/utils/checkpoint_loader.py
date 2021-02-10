import torch
import torch.nn.modules

def load_pretrained_weight(model: torch.nn.modules, weight_path: str, device: torch.device, classifier_name: str = "classifier", filter_fn=None):

    state_dict = torch.load(weight_path, map_location=device)

    strict = True
    classifier_name = classifier_name

    # completely discard fully connected for all other differences between pretrained and created model
    for module_name in state_dict:
        if classifier_name in module_name:
            del state_dict[module_name]

    strict = False

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)