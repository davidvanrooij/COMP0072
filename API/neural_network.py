#Importing libraries.
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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

                if batch_index % 300 == 299:
                    
                    print('[Epoch: %d Batch: %5d] loss: %.3f' % 
                          (epoch + 1, batch_index+1, Current_loss/300)) 
                    
                    self.Losses.append(Current_loss/300)
                    self.Accuracies.append(Current_accuracy/300)

                    Current_loss = 0.0
                    Current_accuracy = 0.0
      
        print('Training finished')
        
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

    fig = plt.figure(figsize=plt.figaspect(0.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(Training_loss,'r')
    plt.ylabel('Average loss per batch')
    ax1.axes.get_xaxis().set_ticks([])
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(Training_accuracy)
    plt.ylabel('Average accuracy per batch')
    ax1.axes.get_xaxis().set_ticks([])
    plt.show()
    
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


    # Save NN to file
    torch.save(net, 'tensor.pt')
        
    print('\n\n############ FINISH ############')


