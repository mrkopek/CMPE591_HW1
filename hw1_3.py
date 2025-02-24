from email.mime import image
from turtle import color
import numpy as np

import torch

from homework1 import Hw1Env, collect
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, random_split
from torch.nn import functional as F
class MyDataset(Dataset):
    def __init__(self,images_before, images_after, actions):
        
        self.images_before = torch.load(images_before).float()
        self.images_after = torch.load(images_after).float()
        self.actions = torch.load(actions).int()
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        action = torch.Tensor(np.zeros([4]))
        action[self.actions[idx]] = 1
        image_before = self.images_before[idx]
        image_after = self.images_after[idx]

        return image_before, action, image_after


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
#self.fc1 = torch.nn.Linear(32 * 30*30 + 4, 128)
        #self.fc2 = torch.nn.Linear(128, 32 * 99 * 99)
        self.deconv1 = torch.nn.ConvTranspose2d(36, 16, 36)
        self.deconv2 = torch.nn.ConvTranspose2d(16, 3, 64)

    def forward(self, img, action):
        x = self.conv1(img)
        x = self.maxpool(F.relu(x))
        x = self.conv2(x)
        x = self.maxpool(F.relu(x))
        
        # Expand action vector to match spatial dimensions of feature maps
        action_expanded = action.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        
        # Concatenate along the channel dimension
        x = torch.cat((x, action_expanded), dim=1)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))

        return x

def train():
    train_loss =[]
    val_loss = []
    show_image = False
    actions_file = "/home/alperen/cmpe591.github.io/data/hw1/actions_0.pt"
    images_before_file = "/home/alperen/cmpe591.github.io/data/hw1/imgs_before_0.pt"
    images_after_file = "/home/alperen/cmpe591.github.io/data/hw1/imgs_after_0.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MyModel().to(device)
    dataset = MyDataset(images_before_file, images_after_file, actions_file)

    # Split dataset into training and validation sets
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    traindataset, validdataset = random_split(dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=8)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=1)

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    train_line, = ax.plot(train_loss, label='Training Loss', color='blue')
    val_line, = ax.plot(val_loss, label='Validation Loss', color='red')
    ax.legend()
    ax.title.set_text('Training and Validation Losses of Deconv')
    # Training loop
    for epoch in range(500):
        model.train()
        running_loss = 0.0
        for imgs_before, actions, imgs_after in trainloader:
            imgs_before, actions, imgs_after = imgs_before.to(device), actions.to(device), imgs_after.to(device)
            optimizer.zero_grad()
            imgs_before = imgs_before.float()
            outputs = model(imgs_before, actions)

            loss = criterion(outputs, imgs_after)

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
            for imgs_before, actions, imgs_after in validloader:
                
                imgs_before, actions, imgs_after = imgs_before.to(device), actions.to(device), imgs_after.to(device)
                imgs_before = imgs_before.float()
                outputs = model(imgs_before, actions)
                
                loss = criterion(outputs, imgs_after)
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
        if show_image:
            show_image_f(outputs[0].squeeze(), title="Output Image")
    plt.savefig("loss_deconv.png")
    torch.jit.save(torch.jit.script(model), "hw1_3.pt")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    action_file = "/home/alperen/cmpe591.github.io/data/hw1/actions_0.pt"
    image_before_file = "/home/alperen/cmpe591.github.io/data/hw1/imgs_before_0.pt"
    image_after_file = "/home/alperen/cmpe591.github.io/data/hw1/imgs_after_0.pt"

    model = torch.jit.load("hw1_3.pt").to(device)
    testdataset = MyDataset(image_before_file, image_after_file, action_file)
    testloader = torch.utils.data.DataLoader(testdataset)

    criterion = torch.nn.MSELoss() 

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs_before, actions, imgs_after in testloader:
            imgs_before, actions, imgs_after = imgs_before.to(device), actions.to(device), imgs_after.to(device)
            imgs_before = imgs_before.float()
            outputs = model(imgs_before, actions)

            loss = criterion(outputs, imgs_after)
            running_loss += loss.item()
        avg_loss = running_loss / len(testloader)
        print(f"Test Loss: {avg_loss:.4f}")

def normalize_image(image_tensor):
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy array and transpose dimensions
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    return image

def show_image_f(image_tensor, title="Image"):
    image = normalize_image(image_tensor)
    plt.figure("Image")
    
    plt.imshow(image)
    plt.title(title)
    plt.show()

def show_output():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    actions_file = "/home/alperen/cmpe591.github.io/data/hw1/actions_0.pt"
    images_before_file = "/home/alperen/cmpe591.github.io/data/hw1/imgs_before_0.pt"
    images_after_file = "/home/alperen/cmpe591.github.io/data/hw1/imgs_after_0.pt"

    model = torch.jit.load("hw1_3.pt").to(device)
    dataset = MyDataset(images_before_file, images_after_file, actions_file)

    _, validdataset = random_split(dataset, [int(0.7 * len(dataset)), len(dataset) - int(0.7 * len(dataset))])
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=1)

    model.eval()
    with torch.no_grad():
        for idx, (imgs_before, actions, imgs_after) in enumerate(validloader):
            if idx == 1:
                imgs_before, actions = imgs_before.to(device), actions.to(device)
                imgs_before = imgs_before.float()
                output = model(imgs_before, actions)
                show_image_f(output.squeeze(), title="Output Image")
                show_image_f(imgs_after.squeeze(), title="Actual Image")
                

if __name__ == "__main__":
    #train()
    test()
    #show_output()