import idx2numpy
from matplotlib import pyplot as plt
import torch
from data.MNIST_Info import MNIST_INFO
from models.ConvNet import ConvNet
from models.IntermediateLayerExtractor import IntermediateLayerExtractor


def plot_intermediates(
    save_path: str,
    intermediate_tuples: tuple[torch.Tensor],
    pick_feature_batch=0,
    pick_feature_channel=0,
):
    total_width = 0
    max_h = 0
    for feature in intermediate_tuples:
        # shape BxCxHxW
        b, c, h, w = feature.shape
        total_width += w
        max_h = max(max_h, h)
    total_img = torch.ones((max_h, total_width))
    running_w = 0
    for feature in intermediate_tuples:
        b, c, h, w = feature.shape
        total_img[0:h, running_w : running_w + w] = feature[
            pick_feature_batch, pick_feature_channel, :, :
        ]
        running_w += w
    img = total_img.numpy()
    plt.imsave(save_path, img)


model = ConvNet(
    input_size=(28, 28),
    num_classes=10,
    model_channels=[(1, 32), (32, 64)],
    strides=[1, 2],
    learning_rate=0.01,
    weight_decay=0.1,
)
extractor = IntermediateLayerExtractor(model, ["layers.2.final_activation"])
extractor.print_named_modules()
file_imgs = MNIST_INFO.train_imgs_path
images = idx2numpy.convert_from_file(file_imgs) / 255.0
index = 0
input = torch.tensor(images[index]).unsqueeze(0).float()
input_batched = torch.stack((input, input), 0)
output = extractor.get_intermediate_layers(
    input_batched, add_output=False, add_input=True
)
[print(out.shape) for out in output]
plot_intermediates(
    "/home/domi/ml-training-technique/playground/test.png",
    output,
)
