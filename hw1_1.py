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
        image = image.view(-1)
        position = self.positions[idx]
        input = torch.cat([image,action])

        return input, position


class MyModel(torch.nn.Module):
    def __init__(self, input_features):
        
        super(MyModel, self).__init__()

        self.fc1 = torch.nn.Linear(input_features, out_features = 20)
        self.fc2 = torch.nn.Linear(in_features=20, out_features=2)
        
    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x))
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

    model = MyModel(49156).to(device)
    dataset = MyDataset(actions_file,images_file,positions_file)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    traindataset, validdataset = random_split(dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=16)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=32)


    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    train_line, = ax.plot(train_loss, label='Training Loss', color='blue')
    val_line, = ax.plot(val_loss, label='Validation Loss', color='red')
    ax.legend()
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Losses of MLP')
    # Training loop
    for epoch in range(500):
        model.train()
        running_loss = 0.0
        for input, positions in trainloader:
            input, positions = input.to(device), positions.to(device)
            optimizer.zero_grad()
            input = input.float()
            outputs = model(input)

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
            for input, positions in validloader:
                input, positions = input.to(device), positions.to(device)
                input = input.float()
                outputs = model(input)
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

    torch.jit.save(torch.jit.script(model), "hw1_1.pt")
    plt.savefig("loss_mlp.png")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    actions_file = "/home/alperen/cmpe591.github.io/data/hw1/actions_0.pt"
    images_file = "/home/alperen/cmpe591.github.io/data/hw1/imgs_before_0.pt"
    positions_file = "/home/alperen/cmpe591.github.io/data/hw1/positions_0.pt"

    model = torch.jit.load("hw1_1.pt").to(device)
    testdataset = MyDataset(actions_file,images_file,positions_file)
    testloader = torch.utils.data.DataLoader(testdataset)

    criterion = torch.nn.MSELoss() 

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for input, positions in testloader:
            input, positions = input.to(device), positions.to(device)
            input = input.float()
            outputs = model(input)
            loss = criterion(outputs, positions)
            running_loss += loss.item()
        avg_loss = running_loss / len(testloader)
        print(f"Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    
    #train()
    test()