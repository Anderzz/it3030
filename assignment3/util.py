import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
import imageio
import os


def tile_images(images: np.ndarray,  no_across: int = None, no_down: int = None,
                show: bool = False, file_name: str = None) -> np.ndarray:
    """
    Take as set of images in, and tile them.
    Input images are represented as numpy array with 3 or 4 dims:
    shape[0]: Number of images
    shape[1] + shape[2]: size of image
    shape[3]: If > 1, then this is the color channel

    Images: The np array with images
    no_across/no_down: Force layout of subfigs. If both arte none, we get a "semi-square" image
    show: do plt.show()
    filename: If not None we save to this filename. Assumes it is fully extended (including .png or whatever)
    """

    no_images = images.shape[0]

    if no_across is None and no_down is None:
        width = int(np.sqrt(no_images))
        height = int(np.ceil(float(no_images) / width))
    elif no_across is not None:
        width = no_across
        height = int(np.ceil(float(no_images) / width))
    else:
        height = no_down
        width = int(np.ceil(float(no_images) / height))

    if len(images.shape) < 4:
        images = np.expand_dims(images, axis=-1)
    color_channels = images.shape[3]

    # Rescale
    images = images - np.min(np.min(np.min(images, axis=(1, 2))))
    images = images / np.max(np.max(np.max(images, axis=(1, 2))))

    # Build up tiled representation
    image_shape = images.shape[1:3]
    tiled_image = np.zeros((height * image_shape[0], width * image_shape[1], color_channels), dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        tiled_image[i * image_shape[0]:(i + 1) * image_shape[0], j * image_shape[1]:(j + 1) * image_shape[1], :] = \
            img          # used to be img[:, :, 0]

    plt.Figure()
    if color_channels == 1:
        plt.imshow(tiled_image[:, :, 0], cmap="binary")
    else:
        plt.imshow(tiled_image.astype(float))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    if show is True:
        plt.show()

    # Clean up
    plt.clf()
    plt.cla()

    return tiled_image

def generate_test_images(model, dataloader, device):
    outputs = []
    labels = []
    model.eval()
    with torch.no_grad():
        for image, label in dataloader:
            # pytorch
            num_channels = image.shape[-1]
            image = image.reshape(-1, num_channels, 28, 28)

            image = image.to(device)
            output = model(image)
            # keras
            output = output.reshape(-1, 28, 28, num_channels)

            outputs.append(output)
            labels.append(label)
    return torch.concatenate(outputs), torch.concatenate(labels)




to_pil_image = transforms.ToPILImage()
def image_to_vid(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('./outputs/generated_images.gif', imgs)

def save_reconstructed_images(recon_images, epoch):
    os.makedirs("./outputs", exist_ok=True)
    save_image(recon_images.cpu(), f"./outputs/output{epoch}.jpg")

def save_loss_plot(train_loss, valid_loss):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/loss.png')
    plt.show()