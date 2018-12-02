# Standard scientific Python imports

from skimage import measure
from skimage import io
from skimage.transform import resize
from skimage.filters import inverse

from PIL import Image
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable


#Defining general variables.
Momentum = 0.9
Batch_size = 50
Epochs_number = 5
Learning_rate = 0.001
np.random.seed(3)

#Building the network.
class Neural_Network(nn.Module):
    
    #Definition.
    def __init__(self):
        super(Neural_Network, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.Losses = []
        self.Accuracies = []
     
    #Forward function.    
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        
        return x 
    
    #Training function.
    def train_net(self, Training_set, Epochs_number, Learning_rate, Momentum):
        
        Loss_function = nn.CrossEntropyLoss()
        Optimizer = optim.SGD(net.parameters(), lr = Learning_rate, momentum = Momentum)

        for epoch in range(Epochs_number):

            Current_loss = 0.0
            Current_accuracy = 0.0
            
            for batch_index, training_batch in enumerate(Training_set, 0):
                
                Inputs, Labels = training_batch
                Inputs, Labels = Variable(Inputs), Variable(Labels)

                Optimizer.zero_grad()
                Outputs = self.forward(Inputs)

                Loss = Loss_function(Outputs, Labels)
                Current_loss += Loss.item()
                
                Loss.backward()
                Optimizer.step()
                
                Total_pred = 0
                Correct_pred = 0
                
                for data in training_batch:
                    Images, Labels = training_batch

                    Outputs = self.forward(Variable(Images))
                    Dummy, Pred_labels = torch.max(Outputs.data, 1)
                    
                    Correct_pred += (Pred_labels == Labels).sum().item()
                    Total_pred += Pred_labels.size(0)
                    
                Current_accuracy += (100 * Correct_pred)/Total_pred
                Current_accuracy += (100 * Correct_pred)/Total_pred

                if batch_index % 300 == 299:
                    
                    print('[Epoch: %d Batch: %5d] loss: %.3f' % 
                          (epoch + 1, batch_index+1, Current_loss/300)) 
                    
                    self.Losses.append(Current_loss/300)
                    self.Accuracies.append(Current_accuracy/300)

                    Current_loss = 0.0
                    Current_accuracy = 0.0
      
        print('Training finished')

class ClassifyImage(object):
    percentage_treshold = 0.035
    output_size = 28
    border_size = 10
        
    def __init__(self):
        self.crop_list = []
        self.cropped_images = []
        
    def set_img(self, img):
        self.img = img #[:, :, 0]
        #self.img = np.invert(self.img)
        
    def get_shape(self):
        return self.img.shape
        
    def crop_image(self, img, y_min, height, x_min, width):
        y_min = int(y_min)
        height = int(height)
        x_min = int(x_min)
        width = int(width)

        return img[y_min:y_min+height , x_min:x_min+width,]
    
    
    def find_img_contours(self):
        self.countours = measure.find_contours(self.img, 10)
        
        self.crop_list = []
        
        for n, contour in enumerate(self.countours):

            # Get extreme values of the contours
            self.y_min, self.x_min = np.min([contour], axis=1)[0]
            self.y_max, self.x_max = np.max([contour], axis=1)[0]

            # Compute the width and height a the box surrounding the countrous
            self.width = self.x_max - self.x_min
            self.height = self.y_max - self.y_min
            

            self.crop_list.append({'y_min': self.y_min, 'height': self.height, 'x_min': self.x_min, 'width': self.width})

        # Sort the array in logical order
        self.crop_list = sorted(self.crop_list, key=lambda x: x['x_min']) 

        print('Number of contours found: {0}'.format(len(self.countours)))
             
        
    def apply_cropping(self):
        # Get size of the original image
        original_size_y, original_size_x = self.get_shape()
        self.count_dropped = 0

        self.find_img_contours()
        
        for crop in self.crop_list:    
            
            self.precentage = (crop['height']*crop['width'])/(original_size_y*original_size_x)
            
            # Crop image
            self.cropped_image = self.crop_image(self.img, crop['y_min'], crop['height'], crop['x_min'], crop['width'])

            if(self.precentage < self.percentage_treshold):
                self.count_dropped += 1
                continue

            print('  -  Percentage of crop / entire canvas {0:.2%}'.format(self.precentage))

            # Prepend and append white vertical white lines on each side to make it a square
            if(crop['height'] > crop['width']):
                ny,nx = self.cropped_image.shape

                # Divide the differences between height and width by 2 since there are two lines added each iteration
                for x in range(0, int((crop['height']-crop['width'])/2)):

                    # Add two lines, one on each side
                    self.cropped_image = np.c_[np.zeros(int(ny)), self.cropped_image, np.zeros(int(ny))]

            # Prepend and append white horizontal white lines on each side to make it a square
            if(crop['width'] > crop['height']):
                ny,nx = self.cropped_image.shape

                # Divide the differences between height and width by 2 since there are two lines added each iteration
                for x in range(0, int((crop['width']-crop['height'])/2)):
                    # Add two lines, one on each side
                    self.cropped_image = np.append(np.zeros([1, int(nx)]), self.cropped_image, axis=0)
                    self.cropped_image = np.append(self.cropped_image, np.zeros([1, int(nx)]), axis=0)

            
            self.cropped_images.append(self.cropped_image)

        print('{0} contours dropped'.format(self.count_dropped))
        
        
    def classify(self):
        print(len(self.cropped_images))

        self.list = []
        
        self.apply_cropping()
        net = Neural_Network()
        net.load_state_dict(torch.load('./API/tensor_sd.pt'))
        net.eval()

        for image in self.cropped_images:
            
            
            image = Image.fromarray(image)
            
            transfrom = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(self.output_size - self.border_size),
                transforms.CenterCrop(self.output_size),
                transforms.ToTensor(),
            ])

            img_tensor = transfrom(image)
            img_tensor.unsqueeze_(0)

            img_tensor.shape

            Outputs = net.forward(Variable(img_tensor))
            Dummy, Predicted_labels = torch.max(Outputs.data, 1)
            
            self.list.append(int(Predicted_labels.numpy().max()))
            print('Classified: {0}'.format(Predicted_labels.numpy().max()))

        return self.list