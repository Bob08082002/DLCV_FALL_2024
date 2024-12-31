import sys
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import imageio.v2 as imageio
from PIL import Image



# do the inference & write result mask to output_dir
def infer_and_save_images(model, input_dir, output_dir, transform):

    for filename in os.listdir(input_dir):
        if filename.endswith('_sat.jpg'):
            img_path = os.path.join(input_dir, filename)

            # Load image and mask
            image = imageio.imread(img_path)
            # using Image.fromarray convert array(from imageio) to PIL image. PIL image之後再transform
            image = Image.fromarray(image).convert('RGB') 
            # from PIL image to tensor. and change the shape from 3*512*512 to 1*3*512*512. and put the image on gpu
            img_tensor = transform(image).unsqueeze(0).to(device)

            #inference
            with torch.no_grad():
                model.to(device) # put the model on gpu
                model.eval()
                output = model(img_tensor.to(device))
                output_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy() #output: 1*7*512*512 #output.squeeze():7*512*512 #output_mask: 512*512    

            # Convert the output to an image
            output_img = Image.fromarray(output_mask.astype('uint8'), mode='P')
            output_img.putpalette([
                0, 255, 255,  # Urban
                255, 255, 0,  # Agriculture
                255, 0, 255,  # Rangeland
                0, 255, 0,    # Forest
                0, 0, 255,    # Water
                255, 255, 255,  # Barren
                0, 0, 0       # Unknown
            ])
            output_img = output_img.convert('RGB') # to 3-channel array

            # Save the result
            output_filename = filename.replace('_sat.jpg', '_mask.png')
            output_path = os.path.join(output_dir, output_filename)
            output_img.save(output_path)
            #print(f"Saved segmentation mask to {output_path}")

#cuz deeplabv3 has dict output
class DeepLabV3_ResNet101(nn.Module):
    def __init__(self, n_classes=7):
        super(DeepLabV3_ResNet101, self).__init__()
        self.model = self.change_output_channel(n_classes)
        
    def forward(self, x):
        return self.model(x)['out']

    def change_output_channel(self, n_classes=7):
        # Load pretrained backbone
        model = deeplabv3_resnet101(weights=None)
        model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))  
        return model


if __name__ == '__main__':
    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mean = [0.4085139,  0.37851316, 0.28088593]
    train_std = [0.14234419, 0.10848381, 0.09824713]
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        # data normalization    # standardization: (image - train_mean) / train_std
        transforms.Normalize(train_mean, train_std),
    ])


    #load DeepLabV3_ResNet101
    net = DeepLabV3_ResNet101(n_classes=7)
    #net.load_state_dict(torch.load('checkpoint_model_all_model/P2/P2_B/deeplabv3/best_val_mIoU_model.pth', map_location=torch.device('cpu')), strict=False)
    net.load_state_dict(torch.load('checkpoint_model/P2/P2_B/deeplabv3/best_val_mIoU_model.pth', map_location=torch.device('cpu')), strict=False)
    net = net.to(device)
    net.eval()


    # val transform和test transform一樣#沒有random flipped(p = 0)
    transform_test = transform_val

    # get $1 & $2
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # do the inference & write result mask to output_dir
    infer_and_save_images(net, input_dir, output_dir, transform_test)

    print("Inferece completed")
