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
        base_learning_rate:float --> the base learning rate to which the learning rate factors are matched
        dont_decay_parameters:list[str] --> list of layer names that should not be decayed
        learning_rate_factors: dict[float,str] --> dict of learning rate factors mapping the factors to layer names that should obtain the factor
    """
    assert (
        not dont_decay_parameters is None or not learning_rate_factors is None
    ), "only use this functionality to split parameter groups"
    if dont_decay_parameters and learning_rate_factors:
        params_with_decay = {}
        params_without_decay = {}
        # remap learning_rate
        learning_rate_map = {
            # name: lr
        }
        intersection = {
            # lr:[name] name is special lr and weight decay 0
        }
        for learning_rate_factor, lr_names in learning_rate_factors.items():
            for lr_name in lr_names:
                learning_rate_map[lr_name] = learning_rate_factor
                if lr_name in dont_decay_parameters:
                    # add to intersection
                    if not learning_rate_factor in intersection:
                        intersection[learning_rate_factor] = [lr_name]
                    else:
                        intersection[learning_rate_factor].append(lr_name)
                    # remove from dont_decay_parameters and from learning_rate_factors
                    dont_decay_parameters.remove(lr_name)
                    learning_rate_factors[learning_rate_factor].remove(lr_name)
        for name, param in model.named_parameters():
            for learning_rate_factor, lr_names in intersection.items():
                for lr_name in lr_names:
                    if lr_name in name:
                        if not learning_rate_factor in params_without_decay:
                            params_without_decay[learning_rate_factor] = [param]
                        else:
                            params_without_decay[learning_rate_factor].append(param)
        for name, param in model.named_parameters():
            name_added = False
            for dont_decay_name in dont_decay_parameters:
                if name_added:
                    break
                for learning_rate_factor, lr_names in learning_rate_factors.items():
                    if name_added:
                        break
                    for lr_name in lr_names:
                        if name_added:
                            break

                        if dont_decay_name in name and lr_name in name:
                            raise ValueError(
                                "This param should already be handled in intersection loop"
                            )
                        elif dont_decay_name in name:
                            name_added = True
                            if not 1 in params_without_decay:  # 1 means original lr
                                params_without_decay[1] = [param]
                            else:
                                params_without_decay[1].append(param)
                        elif lr_name in name:
                            name_added = True
                            if not learning_rate_factor in params_with_decay:
                                params_with_decay[learning_rate_factor] = [param]
                            else:
                                params_with_decay[learning_rate_factor].append(param)
            if not name_added:
                is_in_intersection = False
                # check if in intersection
                for name_intersection in learning_rate_map:
                    if name_intersection in name:
                        is_in_intersection = True
                if not is_in_intersection:
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
    elif dont_decay_parameters:
        weights_with_decay = []
        weights_without_decay = []
        for name, param in model.named_parameters():
            name_added = False
            for dont_decay in dont_decay_parameters:
                if name_added:
                    break
                if dont_decay in name:
                    name_added = True
                    weights_without_decay.append(param)
            if not name_added:
                weights_with_decay.append(param)
        params = [
            {"params": weights_with_decay},
            {"params": weights_without_decay, "weight_decay": 0.0},
        ]
    elif learning_rate_factors:
        weights = {}
        for name, param in model.named_parameters():
            name_added = False
            for learning_rate_factor, lr_names in learning_rate_factors.items():
                if name_added:
                    break
                for lr_name in lr_names:
                    if name_added:
                        break
                    if lr_name in name:
                        name_added = True
                        if not learning_rate_factor in weights:
                            weights[learning_rate_factor] = [param]
                        else:
                            weights[learning_rate_factor].append(param)
            if not name_added:
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
        print("-" * 30)
        print("Base Learning Rate:", base_learning_rate)
        for i, group in enumerate(params):
            print("-" * 20)
            print(f"Parameter group pg{i+1}: {len(group["params"])} parameter tensors")
            if "weight_decay" in group and group["weight_decay"] == 0.0:
                print(f"Weight Decay: 0")
            else:
                print("Weight Decay: Inherit from Global")
            if "lr" in group:
                print(f"Learning Rate: {group["lr"]}")
    # assert num params
    expected_num = len(list(model.named_parameters()))
    partitioned_num = 0
    for param_group in params:
        partitioned_num += len(param_group["params"])
    assert (
        expected_num == partitioned_num
    ), f"Params should be partitioned bijectively but distributed {expected_num} params into groups totaling {partitioned_num}."
    return params
