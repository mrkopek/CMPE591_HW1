from turtle import color
import numpy as np

import torch

from homework1 import Hw1Env, collect
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, random_split
from torch.nn import functional as F
class MyDataset(Dataset):
    def __init__(self,actions,images,positions):
        
        self.actions = torch.load(actions).int()
        self.images = torch.load(images).float()
        self.positions = torch.load(positions).float()
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        action = torch.Tensor(np.zeros([4]))
        action[self.actions[idx]] = 1
        image = self.images[idx]
        #image = image.view(-1)
        position = self.positions[idx]

        return image, action, position


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.fc1 = torch.nn.Linear(32 * 30*30 + 4, 128)
        self.fc2 = torch.nn.Linear(128,2)

    def forward(self, img, action):
        x = self.conv1(img)
        x = self.maxpool(F.relu(x))
        x = self.conv2(x)
        x = self.maxpool(F.relu(x))
        x = x.view(-1, 32 * 30*30)
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train():
    train_loss =[]
    val_loss = []
    
    actions_file = "/home/alperen/cmpe591.github.io/data/hw1/actions_0.pt"
    images_file = "/home/alperen/cmpe591.github.io/data/hw1/imgs_before_0.pt"
    positions_file = "/home/alperen/cmpe591.github.io/data/hw1/positions_0.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MyModel().to(device)
    dataset = MyDataset(actions_file,images_file,positions_file)

    # Split dataset into training and validation sets
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    traindataset, validdataset = random_split(dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=16)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=32)

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    train_line, = ax.plot(train_loss, label='Training Loss', color='blue')
    val_line, = ax.plot(val_loss, label='Validation Loss', color='red')
    ax.legend()
    ax.title.set_text('Training and Validation Losses of CNN')
    # Training loop
    for epoch in range(500):
        model.train()
        running_loss = 0.0
        for imgs, actions, positions in trainloader:
            imgs, actions, positions = imgs.to(device), actions.to(device), positions.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, actions)


            loss = criterion(outputs, positions) 

            loss.backward()

            optimizer.step()

           
            running_loss += loss.item()

        
        train_loss.append(running_loss / len(trainloader))
        train_line.set_ydata(train_loss)
        train_line.set_xdata(range(len(train_loss)))
        train_line.set_color('blue')

        

        print(f"Epoch {epoch+1}, Loss: {train_loss[epoch]:.4f}")

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, actions, positions in validloader:
                images, actions, positions = images.to(device), actions.to(device), positions.to(device)
                images = images.float()
                outputs = model(images, actions)

                loss = criterion(outputs, positions)
                running_loss += loss.item()
        val_loss.append(running_loss / len(validloader))
        val_line.set_ydata(val_loss)
        val_line.set_xdata(range(len(val_loss)))
        val_line.set_color('red')

        ax.set_xlim(0, epoch)
        ax.set_ylim(1e-3, 1e10)
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot

        print(f"Validation Loss: {val_loss[epoch]:.4f}")
    plt.savefig("loss_cnn.png")
    torch.jit.save(torch.jit.script(model), "hw1_2.pt")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    action_file = "/home/alperen/cmpe591.github.io/data/hw1/actions_0.pt"
    image_file = "/home/alperen/cmpe591.github.io/data/hw1/imgs_before_0.pt"
    position_file = "/home/alperen/cmpe591.github.io/data/hw1/positions_0.pt"

    model = torch.jit.load("hw1_2.pt").to(device)
    testdataset = MyDataset(action_file,image_file,position_file)
    testloader = torch.utils.data.DataLoader(testdataset)

    criterion = torch.nn.MSELoss() 

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images,actions, positions in testloader:
            images,actions, positions = images.to(device),actions.to(device), positions.to(device)
            images = images.float()
            outputs = model(images, actions)
            
            loss = criterion(outputs, positions)
            running_loss += loss.item()
        avg_loss = running_loss / len(testloader)
        print(f"Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    
    #train()
    test()