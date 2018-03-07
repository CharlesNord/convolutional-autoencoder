import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

train_set = torchvision.datasets.MNIST(root='../mnist', train=True, transform=transforms.ToTensor(), download=False)
test_set = torchvision.datasets.MNIST(root='../mnist', train=False, transform=transforms.ToTensor(), download=False)

train_loader = Data.DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(dataset=test_set, batch_size=50, shuffle=True, num_workers=4)


class CAE(torch.nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),  # 14*14
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),  # 7*7
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),  # 3*3
            torch.nn.ReLU()
        )
        self.embedding = torch.nn.Linear(in_features=3 * 3 * 128, out_features=10)
        self.fc = torch.nn.Linear(in_features=10, out_features=128 * 3 * 3)  # 3*3
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0),  # 7*7
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2,
                                     output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, stride=2, padding=2,
                                     output_padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        feat = self.embedding(out.view(-1, 3 * 3 * 128))
        out = self.fc(feat)
        out = self.decoder(out.view(-1, 128, 3, 3))
        return out, feat


loss_func = torch.nn.BCELoss()
model = CAE().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data).cuda()
        optimizer.zero_grad()
        recon_batch, _ = model(data)
        loss = loss_func(recon_batch, data)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)
            ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = Variable(data, volatile=True).cuda()
        recon_batch, _ = model(data)
        test_loss += loss_func(recon_batch, data).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(50, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(), './results/reconstruction_' + str(epoch) + '.png', nrow=n)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, 11):
    train(epoch)
    test(epoch)
