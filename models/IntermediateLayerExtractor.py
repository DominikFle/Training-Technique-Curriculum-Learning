from __future__ import annotations
import torch
from torch.nn import ModuleList
import torch.nn as nn


class IntermediateLayerExtractor:
    def __init__(self, model: nn.Module, layer_names: list[str] = None, verbose=False):
        """
        This class can be used to record the features of the model depending on the feature names
        Args:
            model: the model to record the features from
        """
        super().__init__()
        self.model: nn.Module | None = model
        self.__layer_names = layer_names
        self.__recordings = []
        self.__hooks = {}
        self.__device = None
        self.verbose = verbose

    def print_named_modules(self):
        """
        Prints all possible modules, that can be used to extract the feature map from.
        """
        for name, module in self.model.named_modules():
            print(name)

    def __hook(self, _, input: tuple[torch.Tensor], output: torch.Tensor):
        # Take the output of the layer and append it to the recordings
        self.__recordings.append(output.clone().detach().cpu())

    def register_hook(self):
        self.__register_hook()

    def __register_hook(self):
        for name, module in self.model.named_modules():
            if name in self.__layer_names:
                if self.verbose:
                    print("Registered", name)
                self.__hooks[name] = module.register_forward_hook(self.__hook)

    def __eject(self):
        for name, hook in self.__hooks.items():
            hook.remove()
        self.__hooks = {}
        return

    def __clear(self):
        self.__recordings = []

    def get_intermediate_layers(
        self, input: torch.Tensor, add_output=False, add_input=False
    ) -> tuple[torch.Tensor]:
        """
        Args:
            img: torch.Tensor of shape (batch, channels, height, width). This is the input image, that will be run through the model
        Returns:
            A tensor of shape (batch, layers, height/patch_size, width/patch_size)
        """
        if add_input:
            self.__recordings.append(input.clone().detach().cpu())
        self.__register_hook()
        with torch.no_grad():
            out = self.model(input)
        if add_output:
            self.__recordings.append(out.clone().detach().cpu())
        # move all recordings to one device
        target_device = self.__device if self.__device is not None else input.device
        recordings = tuple(map(lambda t: t.to(target_device), self.__recordings))
        self.__eject()
        self.__clear()
        return recordings
