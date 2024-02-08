import cv2
import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import imageio
from torch import nn
from vit import ViT
from vit_grad_rollout import VITAttentionGradRollout
from imageclassification import prepare_dataloaders, set_seed


def show_mask_on_image(img, mask):
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def plot_attention(img, mask, label, ax):
    ax.imshow(img)
    ax.imshow(show_mask_on_image(img, mask), alpha=0.3)
    ax.axis('off')
    ax.set_title(f'Label: {label}')

def main(model_name='', image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         batch_size=16, discard_ratio=0.5, create_gif=False
    ):

    train_iter, val_iter, test_iter, _, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )
    model.to(device)

    # Load the trained model
    model.load_state_dict(torch.load(f"2_VisualTransformers/models/{model_name}.pth"))
    model.eval()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # Get the attention map
    images, labels = next(iter(test_iter))

    attention_rollout = VITAttentionGradRollout(model, discard_ratio=discard_ratio)

    for i, ax in enumerate(axes.flat):
        image = images[i].unsqueeze(0)
        label = labels[i].unsqueeze(0)
        img = image[0].permute(1, 2, 0).numpy()
        image, label = image.to(device), label.to(device)
        
        mask = attention_rollout(image, label)
        # unnormalise img
        img = img * 0.5 + 0.5
        plot_attention(img, mask, labels[i], ax)
    
    fig.suptitle(f'Attention maps for {model_name} with discard ratio {int(discard_ratio*100)}%')
    plt.tight_layout()
    plt.savefig(f'2_VisualTransformers/models/{model_name}_attention_maps_{int(discard_ratio*100)}.png')
    plt.close()

    if create_gif:
        start = 0.6
        end = 0.99
        num_gif_images = 29
        p = pathlib.Path(f'2_VisualTransformers/gif_images/')
        p.mkdir(parents=True, exist_ok=True)
        for discard_ratio in np.linspace(start, end, num_gif_images):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for i, ax in enumerate(axes.flat):
                image = images[i].unsqueeze(0)
                label = labels[i].unsqueeze(0)
                img = image[0].permute(1, 2, 0).numpy()
                image, label = image.to(device), label.to(device)
                
                mask = attention_rollout(image, label)
                # unnormalise img
                img = img * 0.5 + 0.5
                plot_attention(img, mask, labels[i], ax)
            plt.tight_layout()
            plt.savefig(f'2_VisualTransformers/gif_images/{int(discard_ratio*100)}.png')
            plt.close()
        
        images = []
        for i in np.linspace(start, end, num_gif_images):
            images.append(imageio.imread(f'2_VisualTransformers/gif_images/{int(i*100)}.png'))
        
        imageio.mimsave(f'2_VisualTransformers/models/{model_name}_attention_maps.gif', images, fps=10)
        # Remove the images
        for file in p.iterdir():
            file.unlink()
        p.rmdir()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main(model_name='nh4_nl4_best_model_e10', batch_size=6, discard_ratio=0.1, create_gif=True)