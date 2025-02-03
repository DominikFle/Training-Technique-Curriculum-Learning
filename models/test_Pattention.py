import torch
from models.TokenFormerViT import Pattention


def test_Pattention():
    b = 1
    N = 2
    n_before = 80
    n_after = 160  # --> doubling the pattention
    d1 = 96
    d2 = 64
    pattention = Pattention(input_dim=d1, output_dim=d2, param_token_nums=n_before)

    input = torch.rand(1, N, d1)
    print("Pattention key param", pattention.key_param.shape)
    print("Pattention val param", pattention.key_param.shape)
    out = pattention(input)

    assert out.shape == (b, N, d2)

    state_dict = pattention.state_dict()
    for key, val in state_dict.items():
        print(key)
        # increase the key and val params
        shape = val.shape
        assert shape[0] == n_before
        new_param = torch.zeros((n_after, shape[1]))
        new_param[: shape[0], : shape[1]] = val
        state_dict[key] = new_param
    # apply state_dict to new
    pattentionBig = Pattention(input_dim=d1, output_dim=d2, param_token_nums=n_after)
    pattentionBig.load_state_dict(state_dict=state_dict, strict=False)
    print("Pattention key param", pattentionBig.key_param.shape)
    print("Pattention val param", pattentionBig.key_param.shape)
    # Not Working yet, and needs optimizer change as well
    # maybe just restart training all along in a binary fashion n= 100, n=200, n = 400 ..etc

    # apply state_dict to old
    # pattention = Pattention(input_dim=d1, output_dim=d2, param_token_nums=n_after)
    pattention.load_state_dict(state_dict=state_dict, strict=False)
    print("Pattention key param", pattentionBig.key_param.shape)
    print("Pattention val param", pattentionBig.key_param.shape)
    out = pattention(input)
    print("Successful")


if __name__ == "__main__":
    test_Pattention()
