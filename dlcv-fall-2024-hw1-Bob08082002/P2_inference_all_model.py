import sys
import os

import torch
import torch.nn as nn
from torchvision import models, transforms


import imageio.v2 as imageio
from PIL import Image



# an untrained VGG16 + FCN32s model
class VGG16_FCN32s(nn.Module):
    def __init__(self, n_classes=7):
        super(VGG16_FCN32s, self).__init__()

        # Load the pretrained VGG16 model
        #vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        # self.features's shape should be 512*16*16 if input image size is 3*512*512 (H/32, W/32)
        #self.features = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.features = models.vgg16(weights=None).features


        # fc6
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=3, padding=1)  #let its ofmap's shape same as ifmap
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, padding=0) #let its ofmap's shape same as ifmap
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # score
        self.score = nn.Conv2d(4096, n_classes, kernel_size=7, padding=3) #let its ofmap's shape same as ifmap
        # ofmap should be 7*16*16

        # Upsampling: 7*16*16 to 7*512*512
        self.upsample = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=32, stride=32, padding=0, bias=False)

    def forward(self, x):
        # x: 3*512*512
        
        # VGG16 features
        x = self.features(x)
        # x: 512*16*16
        
        # Forward pass through fully convolutional layers
        x = self.relu6(self.fc6(x))
        x = self.drop6(x)
        # x: 4096*16*16

        x = self.relu7(self.fc7(x))
        x = self.drop7(x)
        # x: 4096*16*16

        x = self.score(x)
        # x: 7*16*16

        # Upsample to original image size
        x = self.upsample(x)
        # x: 7*512*512
        
        return x



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







if __name__ == '__main__':
    # Parameters
    BETTER_MODEL = 'VGG16_FCN32s'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mean = [0.4085139,  0.37851316, 0.28088593]
    train_std = [0.14234419, 0.10848381, 0.09824713]
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        # data normalization    # standardization: (image - train_mean) / train_std
        transforms.Normalize(train_mean, train_std),
    ])

    #Load the weight to model
    if BETTER_MODEL == 'VGG16_FCN32s':
        #load VGG16 + FCN32s model
        net = VGG16_FCN32s(n_classes=7)
        net.load_state_dict(torch.load('checkpoint_model/P2/P2_A/best_val_mIoU_model.pth', map_location=torch.device('cpu')), strict=False)
        #net.load_state_dict(torch.load('./checkpoint_model/P2/P2_A/epoch_1st.pth'), strict=False)
        net = net.to(device)
        net.eval()
    elif BETTER_MODEL == 'DeepLabV3Plus':
        #load DeepLabV3Plus model # not only load the weights, but also load the entire model
        net = torch.load('checkpoint_model/P2/P2_B/deeplabv3plus_focalloss/best_val_mIoU_model.pth', map_location=torch.device('cpu')) #net is on cpu, .pth is on gpu
        net = net.to(device)


    # val transform和test transform一樣#沒有random flipped(p = 0)
    transform_test = transform_val

    # get $1 & $2
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # do the inference & write result mask to output_dir
    infer_and_save_images(net, input_dir, output_dir, transform_test)

    print("Inferece completed")
