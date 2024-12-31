import sys
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd

from PIL import Image


# Create CombinedModel with custom classifier
class CombinedModel(nn.Module):
    def __init__(self, Backbone, num_features, hidden1=512, hidden2=256, num_classes=65, dropout_prob=0.5):
        super(CombinedModel, self).__init__()
        #Backbone
        self.Backbone = Backbone
        # First layer
        self.fc1 = nn.Linear(num_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)  # Batch Normalization for first hidden layer
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)  # Dropout after first hidden layer

        # Second layer
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)  # Batch Normalization for second hidden layer
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)  # Dropout after second hidden layer

        # Final layer (output)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        # Backbone(already replace fc layer by identity)
        x = self.Backbone(x)
        # Forward through the first layer
        x = self.fc1(x)
        x = self.bn1(x)  # Batch normalization
        x = self.relu1(x)
        x = self.dropout1(x)  # Dropout
        
        # Forward through the second layer # Second Last layer(fc layer)
        x = self.fc2(x)
        x = self.bn2(x)  # Batch normalization
        x = self.relu2(x)
        x = self.dropout2(x)  # Dropout

        # Output layer (no activation, used for logits) # Last layer
        x = self.fc3(x)
        return x
    
    def get_second_last_layer_output(self, x):
        # Backbone(already replace fc layer by identity)
        x = self.Backbone(x)
        # Forward through the first layer
        x = self.fc1(x)
        x = self.bn1(x)  # Batch normalization
        x = self.relu1(x)
        x = self.dropout1(x)  # Dropout
        
        # Forward through the second layer (this is where we stop)
        x = self.fc2(x)
        x = self.bn2(x)  # Batch normalization
        x = self.relu2(x)
        x = self.dropout2(x)  # Dropout
        
        # Return the second-to-last layer output
        return x




# do the inference & write result mask to output_dir
def infer_and_save_csv(net, input_csv_file, input_image_folder, result_csv_file, transform_test):

    # Read the test CSV
    test_df = pd.read_csv(input_csv_file)

    # Initialize a list to store predictions
    predictions = []

    # Loop over each row in the test CSV
    for idx, row in test_df.iterrows():
        image_filename = row['filename']
        image_path = os.path.join(input_image_folder, image_filename)

        if os.path.exists(image_path):

            # read image at image_path as PIL image, and convert it to 3 channel(以防萬一)
            RGB_PIL_image = Image.open(image_path).convert("RGB")
            image_tensor = transform_test(RGB_PIL_image).unsqueeze(0) #add batch size dimension, ex: 1 * 3 * 128 * 128 

            # Predict the label
            with torch.no_grad():
                net.to(device) # put the model on gpu
                net.eval()
                output = net(image_tensor.to(device))
                _, predicted_label = torch.max(output, 1)
                predicted_label = predicted_label.item()
                
            # Append the result
            predictions.append([row['id'], row['filename'], predicted_label])
        else:
            print(f"Warning: Image {image_filename} not found!")

    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(predictions, columns=['id', 'filename', 'label'])

    # Save the predictions to CSV
    pred_df.to_csv(result_csv_file, index=False)
    print(f"Predictions saved to {result_csv_file}")







if __name__ == '__main__':
    # ------------- Parameters ------------- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mean = [0.485, 0.456, 0.406]
    train_std = [0.229, 0.224, 0.225]
    transform_val = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])



    # ------------- Load the C set pretrained model -------------
    backbone = models.resnet50(weights=None)  # No pretrained weights
    # Number of input features to the original FC layer
    num_features = backbone.fc.in_features  # Typically 2048 for ResNet50 
    backbone.fc = torch.nn.Identity()  # Remove the final classification layer
    net = CombinedModel(Backbone=backbone, num_features=num_features) #net is on cpu
    net.load_state_dict(torch.load('checkpoint_model/P1/Fine_tune/C/best_val_acc_model.pth', map_location=torch.device('cpu')), strict=False)
    net = net.to(device)
    net.eval()

    # val transform和test transform一樣
    transform_test = transform_val

    # get $1 & $2 & $3
    input_csv_file = sys.argv[1]
    input_image_folder = sys.argv[2]
    result_csv_file = sys.argv[3]

    # do the inference & write result csv file
    infer_and_save_csv(net, input_csv_file, input_image_folder, result_csv_file, transform_test)

    print("Inferece completed")
