import numpy as np
from torchvision import transforms


# need to be customized (因為如果image翻轉或平移，mask也應翻轉或平移)
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return image, mask
    
# need to be customized (因為如果image翻轉或平移，mask也應翻轉或平移)
class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        return image, mask

class ToTensor:
    def __call__(self, image):
        return transforms.ToTensor()(image)

class PILToTensor:
    def __call__(self, image):
        return transforms.PILToTensor()(image)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        return transforms.Normalize(self.mean, self.std)(image)

class CustomTransform:
    def __init__(self, p=0.5, mean=None, std=None):
        self.random_flip_h = RandomHorizontalFlip(p)
        self.random_flip_z = RandomVerticalFlip(p)
        self.to_tensor_image = ToTensor() # divide PIL image value by 255
        self.to_tensor_mask = PILToTensor() # no divide PIL image value
        self.normalize = Normalize(mean, std) if mean is not None and std is not None else None

    def __call__(self, image, mask):
        # (如果image翻轉或平移，mask也應翻轉或平移)
        image, mask = self.random_flip_h(image, mask) # apply flip to corresponding image & mask
        image, mask = self.random_flip_z(image, mask) # apply flip to corresponding image & mask
        

        image = self.to_tensor_image(image)
        if self.normalize:
            image = self.normalize(image)# (mask不能被normalize)
        

        mask = self.to_tensor_mask(mask)
        
        return image, mask
