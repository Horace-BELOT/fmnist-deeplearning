"""
File containing code to print / plot specific component of datasets
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from data import DatasetDownLoader, DatasetTransformer


def plot_fashion_mnist_sample(save_path: str = "figures/samples/fashion_mnist.png"):
    
    # Preparing data
    train_ds, _, _ = DatasetDownLoader.fashion_mnist(validation_ratio=0.2)
    f_transform = transforms.ToTensor()
    train_ds = DatasetTransformer(train_ds, f_transform)
    kwargs = {"batch_size": 128, "shuffle": True, "num_workers": 4}
    train_loader = DataLoader(dataset=train_ds, **kwargs)
    
    nsamples: int = 10
    classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    imgs, labels = next(iter(train_loader))

    fig=plt.figure(figsize=(20, 5), facecolor='w')
    for i in range(nsamples):
        ax = plt.subplot(1,nsamples, i+1)
        plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_fashion_mnist_sample()