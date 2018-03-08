import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.manifold import TSNE

train_data = torchvision.datasets.MNIST(root='../mnist', train=True, transform=transforms.ToTensor(), download=False)
test_data = torchvision.datasets.MNIST(root='../mnist', train=False, transform=transforms.ToTensor(), download=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(dataset=test_data, batch_size=50, shuffle=True, num_workers=4)

code_dim = 10


class CVAE(torch.nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1 + 10, out_channels=32, kernel_size=5, stride=2, padding=2),  # 14*14
            #torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),  # 14*14
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),  # 7*7
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),  # 3*3
            torch.nn.ReLU(),
        )

        self.mu = torch.nn.Conv2d(in_channels=128, out_channels=code_dim, kernel_size=3, stride=1,
                                  padding=0)  # code_dim*1*1
        self.logvar = torch.nn.Conv2d(in_channels=128, out_channels=code_dim, kernel_size=3, stride=1,
                                      padding=0)  # code_dim*1*1

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=code_dim + 10, out_channels=128, kernel_size=3, stride=1, padding=0),
            # 3*3
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2,
                                     output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, stride=2, padding=2,
                                     output_padding=1),
            torch.nn.Sigmoid()
        )

    def encode(self, x, label):
        c = Variable(label_transform(label, x)).cuda()
        h1 = self.encoder(torch.cat([x, c], dim=1))
        # h1 = self.encoder(x)
        return self.mu(h1), self.logvar(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z, label):
        c = Variable(label_transform(label, z)).cuda()
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, label):
        mu, logvar = self.encode(x, label)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z, label)
        return out, mu, logvar


model = CVAE().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= x.numel()
    return KLD + BCE


def scatter(feat, label, epoch):
    if code_dim > 2:
        if feat.shape[0] > 5000:
            feat = feat[:5000, :]
            label = label[:5000]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
        feat = tsne.fit_transform(feat)

    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', 10))
    ax = plt.subplot(aspect='equal')
    # sc = ax.scatter(feat[:, 0], feat[:, 1], lw=0, s=40, c=palette[label.astype(np.int)])
    for i in range(10):
        plt.plot(feat[label == i, 0], feat[label == i, 1], '.', c=palette[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.axis('tight')
    for i in range(10):
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

    plt.draw()
    plt.savefig('./conditional_conv_vae/scatter_{}.png'.format(epoch))
    plt.pause(0.001)


def label_transform(labels, data):
    data_shape = data.data.shape
    if labels.shape[0] != data_shape[0]:
        raise ValueError("The number of input data and labels should be the same")

    label = torch.zeros(data_shape[0], 10, data_shape[2], data_shape[3])
    for i in range(data_shape[0]):
        label[i, labels[i], :, :] = 1
    return label


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data).cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label)
        loss = loss_func(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)))

    avg_loss = train_loss / len(train_loader.dataset)

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    return avg_loss


def test(epoch):
    model.eval()
    test_loss = 0
    feat_total = []
    target_total = []
    for i, (data, target) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)  # volatile=True: require_grad=False

        recon_batch, feat, logvar = model(data, target)
        test_loss += loss_func(recon_batch, data, feat, logvar).data[0]
        feat_total.append(feat.data.cpu().view(-1, code_dim))
        target_total.append(target)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(50, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       './conditional_conv_vae/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    feat_total = torch.cat(feat_total, dim=0)
    target_total = torch.cat(target_total, dim=0)
    # scatter(feat_total.numpy(), target_total.numpy(), epoch)

    return test_loss


test_loss_log = []
train_loss_log = []

for epoch in range(1, 21):
    train_loss = train(epoch)
    test_loss = test(epoch)
    train_loss_log.append(train_loss)
    test_loss_log.append(test_loss)
    # Generate from noise
    z = Variable(torch.randn(100, code_dim, 1, 1)).cuda()
    labels = torch.LongTensor(range(10)).repeat(10)
    sample = model.decode(z, labels)
    save_image(sample.data.view(100, 1, 28, 28), './conditional_conv_vae/conditional_conv_vae_sample_'+str(epoch)+'.png', nrow=10)


plt.plot(train_loss_log, 'r--')
plt.plot(test_loss_log, 'g-')
plt.show()
