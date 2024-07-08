import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.linear(x)
        return out


class FC2Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC2Layer, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class CNN2Layer(nn.Module):
    def __init__(self, in_channels, output_size, data_type, n_feature=6):
        super(CNN2Layer, self).__init__()
        self.n_feature = n_feature
        self.intemidiate_size = 5 if data_type == 'cifar10' else 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        self.fc1 = nn.Linear(n_feature * self.intemidiate_size * self.intemidiate_size, 50)  # 4*4 for MNIST 5*5 for CIFAR10
        self.fc2 = nn.Linear(50, output_size)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.2, inplace=False)
        self.batch_norm_2d = nn.BatchNorm2d(n_feature)
        self.batch_norm = nn.BatchNorm1d(50)

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        # x = self.batch_norm_2d(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.n_feature * self.intemidiate_size * self.intemidiate_size)  # 4*4 for MNIST 5*5 for CIFAR10
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.batch_norm(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        return x

class CNN3LayerMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,5)
        self.conv2 = nn.Conv2d(16,16,5)
        self.conv3 = nn.Conv2d(16,32,5)
        self.linear1 = nn.Linear(32*3*3, 32)
        self.linear2 = nn.Linear(32, 10)
        self.batch_norm_2d_1 = nn.BatchNorm2d(16)
        self.batch_norm_2d_2 = nn.BatchNorm2d(16)
        self.batch_norm_1d = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm_2d_1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.batch_norm_2d_2(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        x = x.view(-1,32*3*3)
        x = F.relu(self.linear1(x))
        x = self.batch_norm_1d(x)
        x = self.dropout2(x)
        x = self.linear2(x)

        return x


class CNN3LayerCifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.2, inplace=False)
        self.dropout3 = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
    

#only for cifar10 because it mnist has different resulution and greyscale images
#TODO: try and add nn.dropout2d() to the conv layers    
class CNN5Layer(nn.Module):
    def __init__(self, input, output):
        super(CNN5Layer, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.2, inplace=False)
        self.dropout3 = nn.Dropout(p=0.2, inplace=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.batch_norm2(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.batch_norm3(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 128)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x



