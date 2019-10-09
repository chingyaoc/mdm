import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from models.model import CNNModel
import numpy as np
from test import test

source_dataset_name = 'mnist'
target_dataset_name = 'mnist_m'
source_image_root = os.path.join('dataset', source_dataset_name)
target_image_root = os.path.join('dataset', target_dataset_name)
cuda = True
cudnn.benchmark = True

lr = 1e-3
batch_size = 128
image_size = 28
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])
img_transform_target = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dataset_source = datasets.MNIST(
    root=source_image_root,
    train=True,
    transform=img_transform_source,
    download=True
)
dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)

train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'mnist_m_train'),
    data_list=train_list,
    transform=img_transform_target
)
dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)


net = CNNModel()

# setup optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    net = net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in net.parameters():
    p.requires_grad = True

# training

for epoch in range(n_epoch):
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0

    while i < len_dataloader:        
        net.train()
        net.zero_grad()
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * 0.1

        # source
        data_source = data_source_iter.next()
        s_img, s_label = data_source
        batch_size = len(s_label)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()

        class_output, domain_outputs  = net(input_data=s_img, alpha=alpha)
        err_s_label = loss_class(class_output, s_label)

        err_s_domain = 0
        for domain_output in domain_outputs:
            err_s_domain += loss_domain(domain_output, domain_label) / len(domain_outputs)

        # target
        data_target = data_target_iter.next()
        t_img, _ = data_target
        batch_size = len(t_img)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()
        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        _, domain_outputs = net(input_data=t_img, alpha=alpha)

        err_t_domain = 0
        for domain_output in domain_outputs:
            err_t_domain += loss_domain(domain_output, domain_label) / len(domain_outputs)

        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        i += 1

    test(net, source_dataset_name, epoch)
    test(net, target_dataset_name, epoch)

print('done')
