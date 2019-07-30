import os
import os.path
import soundfile as sf
import librosa
import numpy as np
import scipy
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

batch_size = 100
epochs = 15
learning_rate = 0.004

drop_out = 0.4
image_size = 28*28
first_layer = 100
second_layer = 50
output_size = 10
AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = sf.read(path)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class GCommandLoader(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return len(self.spects)


class ConvolutionNet(nn.Module):
    def __init__(self):
        super(ConvolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20*37*22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 20*37*22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training')

def validate(model, test_loader):
    model.eval()
    average_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            average_loss += F.nll_loss(output, target, size_average=False).item()
            predict = output.max(1, keepdim=True)[1]
            correct += predict.eq(target.view_as(predict)).sum().item()
    size = len(test_loader.sampler)
    average_loss /= size
    accuracy = 100. * correct / size
    print('Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        average_loss, correct, size, accuracy))
    return [average_loss, accuracy]


def test(model, test_loader):
    real_targets = []
    targets = []
    model.eval()
    average_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            average_loss += F.nll_loss(output, target, size_average=False).item()
            predict = output.max(1, keepdim=True)[1]
            correct += predict.eq(target.view_as(predict)).sum().item()
            targets.append(predict)
            real_targets.append(target)
    size = len(test_loader.sampler)
    average_loss /= size
    accuracy = 100. * correct / size
    print('Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        average_loss, correct, size, accuracy))

    np.savetxt("test.pred", targets, fmt='%d', delimiter='\n')
    np.savetxt("real.pred", real_targets, fmt='%d', delimiter='\n')
    return [average_loss, accuracy]


def create_data():

    dataset = GCommandLoader('train_small')
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    dataset = GCommandLoader('valid_small')
    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False,
        num_workers=20, pin_memory=True, sampler=None)

    dataset = GCommandLoader('test_f')
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False,
        num_workers=20, pin_memory=True, sampler=None)

    return [train_loader, validation_loader, test_loader]


def plot_losses(train_losses, validation_losses, title):
    plt.plot(train_losses, 'r')
    plt.plot(validation_losses, 'b')
    plt.title(title)
    plt.ylabel('Losses: Train - red, Validation - blue')
    plt.xlabel('epoch')
    plt.show()


def main():
    # loading the data
    train_loader, validation_loader, test_loader = create_data()
    # define a Convolutional Neural Network
    model = ConvolutionNet()
    # define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracy = []
    validation_losses = []
    validation_accuracy = []

    # train the network
    for epoch in range(1, epochs + 1):
        loss, accuracy = train(model, train_loader, optimizer, criterion, epoch)
        train_losses.append(loss)
        train_accuracy.append(accuracy)

        loss, accuracy = validate(model, validation_loader)
        validation_losses.append(loss)
        validation_accuracy.append(accuracy)

    test_loss, test_accuracy = test(model, test_loader)

    print("Train:")
    print(np.mean(train_losses))
    print(np.mean(train_accuracy))

    print("Validation:")
    print(np.mean(validation_losses))
    print(np.mean(validation_accuracy))

    print("Test:")
    print(np.mean(test_loss))
    print(np.mean(test_accuracy))

    plot_losses(train_losses, validation_losses, 'Convolution')


if __name__ == '__main__':
    main()