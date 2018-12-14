# Standard scientific Python imports
import os
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from skimage import measure

from PIL import Image

import matplotlib.pyplot as plt

from neural_network import Neural_Network

if 'FLASK_ENV' in os.environ and os.environ['FLASK_ENV'] == 'development':
    TENSOR_LOCATION = 'tensor_letters_sd.pt'
else:
    TENSOR_LOCATION = './API/tensor_letters_sd.pt'


class ClassifyImage():
    output_size = 28
    border_size = 10

    def __init__(self):
        self.crop_list = []
        self.cropped_images = []
        self.to_be_deleted = []
        self.countours = None
        self.img = None
        self.count_dropped = 0


    def set_img(self, img):
        """Set image"""
        self.img = img

    def show_original_img(self):
        """Shows the input image"""
        plt.imshow(self.img, cmap=plt.cm.gray_r)
        plt.show()

    @classmethod
    def crop_image(cls, img, y_min, height, x_min, width):
        """Crops a subsection of an image in array format given the dimensions"""
        y_min = int(y_min)
        height = int(height)
        x_min = int(x_min)
        width = int(width)

        return img[y_min:y_min+height, x_min:x_min+width,]

    def find_img_contours(self, show_output=False):
        """Finds contours in an Image and adds every contour to a crop list"""
        if show_output:
            from matplotlib import colors as mcolors
            color = list(mcolors.BASE_COLORS.values())

        self.countours = measure.find_contours(self.img, 10)
        self.crop_list = []

        for index, contour in enumerate(self.countours):

            # Get extreme values of the contours
            y_min, x_min = np.min([contour], axis=1)[0]
            y_max, x_max = np.max([contour], axis=1)[0]

            # Compute the width and height a the box surrounding the countrous
            width = x_max - x_min
            height = y_max - y_min

            if show_output:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
                plt.imshow(self.img, cmap=plt.cm.gray_r, interpolation='nearest')

                plt.axvline(x=x_min, color=color[index], linestyle='--')
                plt.axvline(x=x_min + width, color=color[index], linestyle='--')
                plt.axhline(y=y_min + height, color=color[index], linestyle='--')
                plt.axhline(y=y_min, color=color[index], linestyle='--')


            self.crop_list.append({
                'y_min': y_min,
                'y_max': y_max,
                'height': height,
                'x_min': x_min,
                'x_max': x_max,
                'width': width
                })

        # Sort the array in logical order
        self.crop_list = sorted(self.crop_list, key=lambda x: x['x_min'])

        # Checks if any item of crop list is an subset of another subset
        # this happens when the inside of an say an 8 or 9 is selected as contour
        # these contours needs to be removed before processing
        for crop in self.crop_list:
            for crop_compare in self.crop_list:
                x_min_within_x_axis = bool(
                    crop['x_min'] > crop_compare['x_min'] and
                    crop['x_min'] < crop_compare['x_max'])

                x_max_within_x_axis = bool(
                    crop['x_max'] < crop_compare['x_max'] and
                    crop['x_max'] > crop_compare['x_min'])

                y_min_within_y_axis = bool(
                    crop['y_min'] > crop_compare['y_min'] and
                    crop['y_min'] < crop_compare['y_max'])

                y_max_within_x_axis = bool(
                    crop['y_max'] < crop_compare['y_max'] and
                    crop['y_max'] > crop_compare['y_min'])

                # crop is subset of another crop when this is true
                if(x_min_within_x_axis and x_max_within_x_axis and
                   y_min_within_y_axis and y_max_within_x_axis):

                    self.to_be_deleted.append(crop)

        # Delete al the items form the to be deleted list
        self.crop_list = [x for x in self.crop_list if x not in self.to_be_deleted]

        print('Number of contours found: {0}'.format(len(self.countours)))
        print('Number of contours deleted: {0}'.format(len(self.to_be_deleted)))
        print('Number in final crop list: {0}'.format(len(self.crop_list)))

        if show_output:
            plt.title('Full image with crop lines')
            plt.show()

    def apply_cropping(self, show_output=False):
        """Loops through the images and crops accordingly. 
           Adds whitespace around the crop to make it a square"""
        self.find_img_contours(show_output=show_output)

        self.cropped_images = []

        for crop in self.crop_list:

            # Crop image
            cropped_image = self.crop_image(self.img, crop['y_min'], crop['height'],
                                            crop['x_min'], crop['width'])

            if show_output:
                # Plot cropped image
                plt.imshow(cropped_image, cmap=plt.cm.gray_r, interpolation='nearest')
                plt.title('Cropped number')
                plt.show()

            # Prepend and append white vertical white lines on each side to make it a square
            if crop['height'] > crop['width']:
                y, x = cropped_image.shape

                # Divide the differences between height and width by 2
                # since there are two lines added each iteration
                for i in range(0, int((crop['height']-crop['width'])/2)):

                    # Add two lines, one on each side
                    cropped_image = np.c_[np.zeros(int(y)),
                                          cropped_image, np.zeros(int(y))]

            # Prepend and append white horizontal white lines on each side to make it a square
            if crop['width'] > crop['height']:
                y, x = cropped_image.shape

                # Divide the differences between height and width by 2
                # since there are two lines added each iteration
                for i in range(0, int((crop['width']-crop['height'])/2)):
                    # Add two lines, one on each side
                    cropped_image = np.append(np.zeros([1, int(x)]),
                                              cropped_image, axis=0)

                    cropped_image = np.append(cropped_image,
                                              np.zeros([1, int(x)]), axis=0)

            if show_output:
                # Plot cropped squared image
                plt.imshow(cropped_image, cmap=plt.cm.gray_r, interpolation='nearest')
                plt.title('Make corpped image a square')
                plt.show()

            self.cropped_images.append(cropped_image)

        print('{0} contours dropped'.format(self.count_dropped))


    def classify(self, show_output=False):
        """Send the preprocessed images to the NN classifier"""
        print('{0} Numbers to be classified'.format(len(self.cropped_images)))

        return_list = []
        self.apply_cropping(show_output=show_output)
        net = Neural_Network()
        net.load_state_dict(torch.load(TENSOR_LOCATION))
        net.eval()

        for image in self.cropped_images:

            image = Image.fromarray(image)

            # Resizes the number and adds a 10 px border
            transfrom = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(self.output_size - self.border_size),
                transforms.CenterCrop(self.output_size),
                transforms.ToTensor(),
            ])

            img_tensor = transfrom(image)

            if show_output:
                plt.imshow(np.array(img_tensor)[0, :, :],
                           cmap=plt.cm.gray_r, interpolation='nearest')
                plt.title('Image used for classification')
                plt.show()

            img_tensor.unsqueeze_(0)

            outputs = net.forward(Variable(img_tensor))
            dummy, predicted_labels = torch.max(outputs.data, 1)

            return_list.append(int(predicted_labels.numpy().max()))
            print('Classified: {0}'.format(predicted_labels.numpy().max()))

        return return_list
