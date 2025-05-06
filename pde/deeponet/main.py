import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse 

from model.deeponet import DeepONetCartesianProd
from data.triple import TripleCartesianProd
# from train import LossHistory, Model, TrainState

def get_data(filename):
    nx = 40
    nt = 40
    data = np.load(filename)
    x = torch.tensor(data["x"].astype(np.float32))
    t = torch.tensor(data["t"].astype(np.float32))
    u = torch.tensor(data["u"].astype(np.float32))  # N x nt x nx

    u0 = u[:, 0, :]  # N x nx
    xt = torch.vstack((x.ravel(), t.ravel())).T
    u = u.reshape(-1, nt * nx)
    return (u0, xt), u


def main():
    parser = argparse.ArgumentParser(description="DeepONet for IC2")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=250000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--model_name', type=str, default='deeponet', help='Name of the model, deeponet or deeponet_kinet')
    args = parser.parse_args()
    device = f'cuda:{0}'
    # Load data
    nt = 40
    nx = 40
    x_train, y_train = get_data("data/train_IC2.npz")
    x_test, y_test = get_data("data/test_IC2.npz")
    data = TripleCartesianProd(x_train, y_train, x_test, y_test)
    inputs_train = data.train_x 
    inputs_test = data.test_x
    truth_train = data.train_y
    truth_test = data.test_y
    inputs_train = tuple(input_.to(device) for input_ in inputs_train)
    inputs_test = tuple(input_.to(device) for input_ in inputs_test)
    truth_train = truth_train.to(device)
    truth_test = truth_test.to(device)

    net = DeepONetCartesianProd([nx, 512, 512], [2, 512, 512, 512])
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.1)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(args.epochs):
        net.train()
        optimizer.zero_grad()
        y_pred = net(inputs_train)
        loss = loss_fn(y_pred, truth_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0:
            net.eval()
            with torch.no_grad():
                y_pred = net(inputs_test)
                test_loss = loss_fn(y_pred, truth_test)
                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    torch.save(net.state_dict(), "log_resnet/best_model.pth")
                print(f"Epoch {epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}, Best Loss: {best_loss}")
                print(' ')

    net.load_state_dict(torch.load("log/best_model.pth"))
    net.eval()
    with torch.no_grad():
        y_pred = net(inputs_test)
    y_pred = y_pred.cpu().numpy()
    y_test = data.test_y[0].cpu().numpy()
    np.savetxt("log_resnet/y_pred_deeponet.dat", y_pred[0].reshape(nt, nx))
    np.savetxt("log_resnet/y_true_deeponet.dat", y_test.reshape(nt, nx))
    np.savetxt("log_resnet/y_error_deeponet.dat", (y_pred[0] - y_test).reshape(nt, nx))


if __name__ == "__main__":
    main()