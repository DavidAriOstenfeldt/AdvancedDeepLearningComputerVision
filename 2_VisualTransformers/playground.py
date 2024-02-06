from einops import rearrange, repeat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
​
from imageclassification import prepare_dataloaders, set_seed
from vit import ViT, positional_encoding_2d
​
def image_to_patches(image, patch_size, image_grid=True):
    """
        image : torch.Tensor of shape B x C x H x W
        patch_size : number of pixels per dimension of the patches (tuple)
        image_grid : If True, the patches are returned as an image grid.
                     else they are return as a sequence of flattened patches
                     B x num_patches x (patch_h * patch_w * C)
    """
    # image shape: B x C x H x W 
    # same as B x C x (num_patches_h * patch_h) x (num_patches_w * patch_w)
    # same as B x C x (num_patches_h * patch_h) x (num_patches_w * patch_w)
    # where num_patches_h = (H // patch_h) and num_patches_w = (W // patch_w)
    # and num_patches = num_patches_h * num_patches_w
    B, C, H, W = image.shape        
    patch_h, patch_w = patch_size
    assert H % patch_h == 0 and W % patch_w == 0, 'image dimensions must be divisible by the patch size.'
    num_patches = (H // patch_h) * (W // patch_w)
​
    if image_grid:
        print(f'number of patches: {num_patches}')
        # B x num_patches x c x patch_h x patch_w
        patches = rearrange(image, 'b c (nph ph) (npw pw) -> b (nph npw) c ph pw', ph=patch_h, pw=patch_w) 
    else:
        # TASK: Implement images to patches functionality 
        #       that returns a sequence of flattened patches
        # HINT: B x num_patchs x ph * pw * c (flatten patches!)
        ####### insert code here #######
        # 
        ################################
        assert patches.size()== (batch_size, num_patches , (patch_h * patch_w * C))
    return patches
​
if __name__ == "__main__":
    # select 2 classes from CIFAR10
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 
    batch_size = 8
    classes = [3,7]
    _, _, dataset, _ = prepare_dataloaders(batch_size=batch_size, classes=classes)
​
    # visualize examples
    example_images = torch.stack([dataset[idx][0] for idx in range(batch_size)], dim=0)
    img_grid = torchvision.utils.make_grid(example_images, nrow=batch_size, 
                                        normalize=True, pad_value=0.9
    )
    img_grid = img_grid.permute(1, 2, 0)
    plt.figure(figsize=(8,8))
    plt.title("Image examples from CIFAR10 dataset")
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()
​
    # convert images to patches
    img_patches = image_to_patches(example_images, patch_size=(4,4), image_grid=True)
​
    # visualize patches
    fig, ax = plt.subplots(example_images.shape[0], 1, figsize=(14,3))
    fig.suptitle("Images as input sequences of patches")
    for i in range(example_images.shape[0]):
        img_grid = torchvision.utils.make_grid(img_patches[i], 
                                            nrow=img_patches.shape[1], 
                                            normalize=True, pad_value=0.9
        )
        img_grid = img_grid.permute(1, 2, 0)
        ax[i].imshow(img_grid)
        ax[i].axis('off')
    plt.show()
​
    # test input patches before linear embedding layer
    img_patches = image_to_patches(example_images, patch_size=(4,4), image_grid=False)
    print(f'shape of input images (batch):  {example_images.shape}')
    print(f'shape of input patches BEFORE the linear projection layer (batch): {img_patches.shape}')
​
    # NOTE: DOES THIS PLOT PROVIDE ANY USEFUL INSIGHT?
    # plot positional_encoding_2d
    H, W = (32, 32)
    patch_h , patch_w = (4,4)
​
    pos_embedding = positional_encoding_2d(
                nph = H // patch_h,
                npw = W // patch_w,
                dim = 128,
            ) 
​
    pos_emb = pos_embedding.view((H // patch_h, W // patch_w,128))
    fig, axes = plt.subplots(8, 16, figsize=(16, 8))
    fig.suptitle('Positional Embedding in each dimension')
​
    for i in range(8):
        for j in range(16):
            dimension = i * 16 + j
            axes[i, j].imshow(pos_emb[:, :, dimension], cmap='viridis')
            axes[i, j].set_title(f'Dim {dimension}')
            axes[i, j].axis('off')
​
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig('pos_embeddings_2.png', bbox_inches='tight', pad_inches=0)
    plt.show()
