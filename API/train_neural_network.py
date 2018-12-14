import torchvision
import torch
import numpy as np

from neural_network import Neural_Network
from torch.autograd import Variable

import torchvision.transforms as transforms
        

#Defining general variables.
Momentum = 0.9
Batch_size = 50
Epochs_number = 5
Learning_rate = 0.001
np.random.seed(3)


#Importing data and splitting train and test set.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#Mean and std for general coulored (3 channels) images.
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=Batch_size, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=Batch_size, shuffle=True)

#Takes the name of the folder as label.
#Data = torchvision.datasets.ImageFolder('path/to/imagenet_root/', transform=transform)
#Data_loader = torch.utils.data.DataLoader(Data, batch_size=Batch_size, shuffle=True)

#Main program.
if __name__ == "__main__":
    
    #Neural network.
    net = Neural_Network()
    net.train_net(train_set_loader, Epochs_number, Learning_rate, Momentum)
    
    #Understanding loss and accurancy.
    Training_loss = net.Losses
    Training_accuracy = net.Accuracies

    # fig = plt.figure(figsize=plt.figaspect(0.2))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.plot(Training_loss,'r')
    # plt.ylabel('Average loss per batch')
    # ax1.axes.get_xaxis().set_ticks([])
    # ax1 = fig.add_subplot(1, 2, 2)
    # ax1.plot(Training_accuracy)
    # plt.ylabel('Average accuracy per batch')
    # ax1.axes.get_xaxis().set_ticks([])
    # plt.show()
    
    #Testing the network.
    Total_pred = 0
    Correct_pred = 0
    Class_total = list(0. for i in range(10))
    Class_correct = list(0. for i in range(10))
    
    for test_data in test_set_loader:
        Test_images, Test_labels = test_data
        
        Outputs = net.forward(Variable(Test_images))
        Dummy, Predicted_labels = torch.max(Outputs.data, 1)

        Correct_pred += (Predicted_labels == Test_labels).sum()
        Total_pred += Predicted_labels.size(0)
        
        Correct = (Predicted_labels == Test_labels).squeeze()
        for i in range(10):
            Label = Test_labels[i]
            Class_correct[Label] += Correct[i]
            Class_total[Label] +=1

    print('Accuracy of the network on the 10000 test images: %d %%' 
          % (100*Correct_pred/Total_pred))

    for i in range(10):
        print('Accuracy of digit %d : %2d %%' % (i, 100*Class_correct[i].item()/Class_total[i]))


    # Save NN to tensor.pt
    torch.save(net, 'tensor.pt')
        
    print('\n\n############ FINISH ############')
