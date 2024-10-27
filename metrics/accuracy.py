import torch


def accuracy(model_out: torch.Tensor, targets: torch.Tensor):
    """
    Takes tensors of size (B,) where for every b in B a class is expected
    """
    model_out = model_out.long()
    targets = targets.long()  # target --> Bx10
    accuracy = torch.sum(model_out == targets) / targets.shape[0]
    return accuracy


def accuracy_from_out_probabilities(
    model_out: torch.Tensor, targets: torch.Tensor, individual_classes=[]
) -> int | tuple[int, list[int]]:
    """
    model_out: B x num_classes      # probabilities
    targets:  B                     # class indices
    """
    out_sharp = torch.argmax(model_out, -1).long()
    targets = targets.long()  # target --> Bx10
    accuracy = torch.sum(out_sharp == targets) / targets.shape[0]
    out = []
    for class_index in individual_classes:
        num_i = torch.sum(targets == class_index)
        targets_i = torch.where(targets == class_index, targets, -1)
        accuracy_i = torch.sum(out_sharp == targets_i) / num_i
        out.append(accuracy_i)

    return accuracy, out
