import chainer
import os
import numpy as np
from PIL import Image

# http://mizti.hatenablog.com/entry/chainer_dataset


class ImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root='.', max_size=32, normalize=True, flatten=False, mean=None):
        self._max_size = max_size
        self._normalize = normalize
        self._flatten = flatten
        self._mean = mean
        pairs = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(' ')
                if len(line) == 2:
                    line[0] = os.path.join(root, line[0])
                    pairs.append(line)
        self._pairs = pairs
        if self._mean:
            self._mean = np.load(self._mean)  # 0.0-255.0

    def __len__(self):
        return len(self._pairs)

    def read_image(self, filename):
        image = Image.open(filename)
        new_w = self._max_size
        new_h = self._max_size
        if new_w != image.size[0] or new_h != image.size[1]:
            image = image.resize((new_w, new_h))
        image_array = np.asarray(image).astype(np.float32)
        return image_array

    def get_example(self, i):
        filename = self._pairs[i][0]
        image_array = self.read_image(filename)
        if self._normalize:
            image_array = image_array / 255
        if self._flatten:
            image_array = image_array.flatten()
        else:
            if image_array.ndim == 2:
                image_array = image_array[np.newaxis, :]
        image_array = image_array.transpose(2, 0, 1)  # (H, W, D) -> (D, H, W)
        if self._mean is not None:
            if self._normalize:
                image_array -= self._mean / 255
            else:
                image_array -= self._mean
        label = np.float32(self._pairs[i][1])
        label = label * 9.0 + 1.0 # rescale to 1.0-10.0
        return image_array, np.array([label], dtype=np.float32)


if __name__ == '__main__':
    dataset = ImageDataset('test.txt', root='data_32x32')
    # dataset = ImageDataset('test.txt', root='data_32x32', mean='mean.npy')
    img, label = dataset[0]
    print('img=', img.shape, 'label=', label)
    print(img)
    Image.fromarray((img * 255).astype(np.uint8).transpose(1, 2, 0)).save('test.jpg')
