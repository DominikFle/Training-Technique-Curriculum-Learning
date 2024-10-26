import torch.nn as nn


class DynamicDropout(nn.Module):
    def __init__(
        self,
        p_storage,  # expects reference to variable like array
        dropout_dim="1d",  # 2d
    ):
        super().__init__()
        assert (
            dropout_dim == "1d" or dropout_dim == "2d"
        ), f"Only 1d or 2d dropout is dynamically supported"
        self.p_storage = p_storage
        self.drop_out_func = (
            nn.functional.dropout if dropout_dim == "1d" else nn.functional.dropout2d
        )

    def forward(self, input):
        return self.drop_out_func(input, self.p_storage[0], self.training)
