import torch.nn as nn


def create_optimizer_groups(
    model: nn.Module,
    base_learning_rate: float,
    dont_decay_parameters: list[str],
    learning_rate_factors: dict[float, list[str]],  # factor --> list[layer names]
    verbose=False,
) -> list[dict]:
    """
    Determines the parameter groups to be used in the optimizers when the grouping gets more complicated.
    Args:
        model: nn.Module --> the model that contains the parameters
        base_learning_rate:flaot --> the base learning rate to which the learning rate factors are matched
        dont_decay_parameters:list[str] --> list of layer names that should not be decayed
        learning_rate_factors: dict[float,str] --> dict of learning rate factors mapping the factors to layer names that should obtain the factor
    """
    assert (
        not dont_decay is None or not learning_rate_factors is None
    ), "only use this functionality to split parameter groups"
    if dont_decay and learning_rate_factors:
        params_with_decay = {}
        params_without_decay = {}
        for name, param in model.named_parameters():
            for dont_decay_name in dont_decay_parameters:
                for learning_rate_factor, lr_name in learning_rate_factors.items():
                    if dont_decay_name in name and lr_name in name:
                        if not learning_rate_factor in params_without_decay:
                            params_without_decay[learning_rate_factor] = [param]
                        else:
                            params_without_decay[learning_rate_factor].append(param)
                    elif dont_decay_name in name:
                        if not 1 in params_without_decay:  # 1 means original lr
                            params_without_decay[1] = [param]
                        else:
                            params_without_decay[1].append(param)
                    elif lr_name in name:
                        if not learning_rate_factor in params_with_decay:
                            params_with_decay[learning_rate_factor] = [param]
                        else:
                            params_with_decay[learning_rate_factor].append(param)
                    else:
                        if not 1 in params_with_decay:
                            params_with_decay[1] = [param]
                        else:
                            params_with_decay[1].append(param)
        params = []
        for lr_factor, weights in params_with_decay.items():
            params.append({"params": weights, "lr": base_learning_rate * lr_factor})
        for lr_factor, weights in params_without_decay.items():
            params.append(
                {
                    "params": weights,
                    "lr": base_learning_rate * lr_factor,
                    "weight_decay": 0.0,
                }
            )
    elif dont_decay:
        weights_with_decay = []
        weights_without_decay = []
        for name, param in model.named_parameters():
            for dont_decay in dont_decay_parameters:
                if dont_decay in name:
                    weights_without_decay.append(param)
                else:
                    weights_with_decay.append(param)
        params = [
            {"params": weights_with_decay},
            {"params": weights_without_decay, "weight_decay": 0.0},
        ]
    elif learning_rate_factors:
        weights = {}
        for name, param in model.named_parameters():
            for learning_rate_factor, lr_name in learning_rate_factors.items():
                if lr_name in name:
                    if not learning_rate_factor in weights:
                        weights[learning_rate_factor] = [param]
                    else:
                        weights[learning_rate_factor].append(param)
                else:
                    if not 1 in weights:
                        weights[1] = [param]
                    else:
                        weights[1].append(param)
        params = []
        for lr_factor, weights in weights.items():
            params.append({"params": weights, "lr": base_learning_rate * lr_factor})
    else:
        ValueError("Should not happen!")
    if verbose:
        for i, group in enumerate(params):
            print(f"Parameter group {i}:")
        if group["weight_decay"] == 0.0:
            print(f"Weight Decay: 0")
        if "lr" in group:
            print(f"Learning Rate: {group["lr"]}")
    return params
