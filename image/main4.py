'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import wandb

from kinet_model import *
import model as model
import model2 as model2
# import model3 as model3
import model4 as model4
from utils import progress_bar, set_seed, apply_gradient_collision

if_wandb = True

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--model', type=str, default="model_kinet18_4")

args = parser.parse_args()

set_seed(args.seed)

device = f'cuda:{args.gpu}' 
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# set up wandb
if if_wandb:
    wandb.init(project="kinet-cifar100", config=args, name=f"{args.model}")

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='../collision/data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='../collision/data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')

if args.model == "resnet18":
    net = ResNet18()
elif args.model == "resnet50":
    net = ResNet50()
elif args.model == "kinet18":
    net = Kinet18()
elif args.model == "newkinet18":
    net = New_Kinet18()
elif args.model == "2kinet18":
    net = Bi_Kinet18()
elif args.model == "3kinet18":
    net = Tri_Kinet18()  
elif args.model == "model_kinet18":
    net = model.KINET_ResNet18()
elif args.model == "test":
    net = model.ResNet18()
elif args.model == "model_kinet18_2":
    net = model2.KINET_ResNet18_2()
# elif args.model == "model_kinet18_3":
#     net = model3.KINET_ResNet18_3()  
elif args.model == "model_kinet18_4":
    net = model4.KINET_ResNet18_4()  
else:
    raise NotImplementedError

net = net.to(device)
print(net)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.collision_type}_{args.collision_rate}_noise_{args.noise_scale}/{args.start_epoch}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch, vec_train_norm_1, vec_train_norm_2, vec_train_norm_3, vec_train_norm_4):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # bs, chnl, h, w = inputs.shape # batch size 
        # n_particles = h * w # number of particles

        # angles = 2 * torch.pi * torch.rand((inputs.shape[0], 512 - 1, 16, 16)).to(inputs.device)
        # angles = torch.zeros((inputs.shape[0], 512 - 1, 16, 16)).to(device)

        optimizer.zero_grad()
        outputs = net(inputs, vec_train_norm_1, vec_train_norm_2, vec_train_norm_3, vec_train_norm_4)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.* correct / total, correct, total))
    if if_wandb:     
        wandb.log({
                'epoch': epoch,
                'train/acc': 100. * correct / total,
                'train/loss': train_loss / len(trainloader),
            })


def test(epoch, vec_test_norm_1, vec_test_norm_2, vec_test_norm_3, vec_test_norm_4):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = net(inputs, vec_test_norm_1, vec_test_norm_2, vec_test_norm_3, vec_test_norm_4)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100. * correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir(f'checkpoint/{args.model}'):
            os.mkdir(f'checkpoint/{args.model}')

        torch.save(state, f'./checkpoint/{args.model}/{start_epoch}.pth')
        best_acc = acc
    if if_wandb:
        wandb.log({
            'test/acc': acc,
        })

n1 = 16
n2 = 8
n3 = 4
n4 = 2

vec_train_1 = torch.randn(128, 64*n2, int((1024/n1)*(1024/n1-1)/2), device=device)
vec_train_ori_1 = vec_train_1 / torch.norm(vec_train_1, dim=1, keepdim=True)
vec_test_1 = torch.randn(100, 64*n2, int((1024/n1)*(1024/n1-1)/2), device=device)
vec_test_1 = vec_test_1 / torch.norm(vec_test_1, dim=1, keepdim=True)

vec_train_2 = torch.randn(128, 128*n2, int((256/n2)*(256/n2-1)/2), device=device)
vec_train_ori_2 = vec_train_2 / torch.norm(vec_train_2, dim=1, keepdim=True)
vec_test_2 = torch.randn(100, 128*n2, int((256/n2)*(256/n2-1)/2), device=device)
vec_test_2 = vec_test_2 / torch.norm(vec_test_2, dim=1, keepdim=True)

vec_train_3 = torch.randn(128, 256*n3, int((64/n3)*(64/n3-1)/2), device=device)
vec_train_ori_3 = vec_train_3 / torch.norm(vec_train_3, dim=1, keepdim=True)
vec_test_3 = torch.randn(100, 256*n3, int((64/n3)*(64/n3-1)/2), device=device)
vec_test_3 = vec_test_3 / torch.norm(vec_test_3, dim=1, keepdim=True)

vec_train_4 = torch.randn(128, 512*n4, int((16/n4)*(16/n4-1)/2), device=device)
vec_train_ori_4 = vec_train_4 / torch.norm(vec_train_4, dim=1, keepdim=True)
vec_test_4 = torch.randn(100, 512*n4, int((16/n4)*(16/n4-1)/2), device=device)
vec_test_4 = vec_test_4 / torch.norm(vec_test_4, dim=1, keepdim=True)

for epoch in range(start_epoch, start_epoch + 200):
    x_1 = -4 + (-7 + 4) * torch.rand(1, device=device)
    sign_1 = torch.randint(0, 2, (1,), device=device) * 2 - 1
    vec_train_1 = vec_train_1 * vec_train_ori_1 + vec_train_ori_1 + sign_1 * torch.exp(x_1)
    vec_train_1 = vec_train_1 / torch.norm(vec_train_1, dim=1, keepdim=True)

    x_2 = -4 + (-7 + 4) * torch.rand(1, device=device)
    sign_2 = torch.randint(0, 2, (1,), device=device) * 2 - 1
    vec_train_2 = vec_train_2 * vec_train_ori_2 + vec_train_ori_2 + sign_2 * torch.exp(x_2)
    vec_train_2 = vec_train_2 / torch.norm(vec_train_2, dim=1, keepdim=True)

    x_3 = -4 + (-7 + 4) * torch.rand(1, device=device)
    sign_3 = torch.randint(0, 2, (1,), device=device) * 2 - 1
    vec_train_3 = vec_train_3 * vec_train_ori_3 + vec_train_ori_3 + sign_3 * torch.exp(x_3)
    vec_train_3 = vec_train_3 / torch.norm(vec_train_3, dim=1, keepdim=True)

    x_4 = -4 + (-7 + 4) * torch.rand(1, device=device)
    sign_4 = torch.randint(0, 2, (1,), device=device) * 2 - 1
    vec_train_4 = vec_train_4 * vec_train_ori_4 + vec_train_ori_4 + sign_4 * torch.exp(x_4)
    vec_train_4 = vec_train_4 / torch.norm(vec_train_4, dim=1, keepdim=True)

    train(epoch, vec_train_1, vec_train_2, vec_train_3, vec_train_4)
    test(epoch, vec_test_1, vec_test_2, vec_test_3, vec_test_4)
    scheduler.step()