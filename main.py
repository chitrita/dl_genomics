import os
import time
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from utils.stats import pearson_correlation
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from models.extractor import get_extractor
from models.regressor import Regressor


random.seed(0)
torch.manual_seed(0)


def str2bool(v):
    """See: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    """
    Main entry point.
    """
    cuda = torch.cuda.is_available()

    X, y = torch.load("./data/geno_oh.pt"), torch.load("./data/pheno.pt").float()

    args.input_length = X.shape[-1]

    extractor = get_extractor(args)
    out = extractor(Variable(X[0:1]))

    if isinstance(out, tuple):
        out, _ = out

    extractor.zero_grad()

    model = Regressor(out.shape[-1] * out.shape[-2], 1, extractor)
    model.initialize_weights()

    # metric = torch.nn.MSELoss()

    n = int(0.8 * X.shape[0])
    permuatation = [i for i in range(X.shape[0])]
    random.shuffle(permuatation)
    testing = permuatation[n:]
    m = int(0.8 * n)
    validation = permuatation[m:n]
    training = permuatation[:m]

    dataloaders = {
        "train": DataLoader(TensorDataset(X, y), batch_size=args.batch_size, sampler=SubsetRandomSampler(training)),
        "valid": DataLoader(TensorDataset(X, y), batch_size=args.batch_size, sampler=SubsetRandomSampler(validation)),
        "test": DataLoader(TensorDataset(X, y), batch_size=args.batch_size, sampler=SubsetRandomSampler(testing))
    }

    if cuda:
        model = model.cuda()
        # metric = metric.cuda()

    optimizer = optim.RMSprop(model.parameters())

    if args.extractor == "sae":
        # Need to train the autoencoder first.

        if os.path.isfile("sae.pt"):
            extractor.load_state_dict(torch.load("sae.pt"))

        else:
            # Train from scratch.
            adam = optim.Adam(extractor.parameters())
            mse = torch.nn.MSELoss()

            if cuda:
                mse = mse.cuda()

            extractor.train()

            for epoch in range(args.epochs):

                start = time.time()

                for batch, _ in dataloaders["train"]:
                    encoded, decoded = extractor(batch)

                    loss = mse(decoded, batch)
                    loss.backward()
                    adam.step()

                    print("Epoch {}: AE Training Loss: {:0.4f}, {:0.4f}s"
                          .format(epoch, loss.item(), time.time() - start))

            torch.save(extractor.state_dict(), "sae.pt")

    try:
        for epoch in range(args.epochs):
            start_time = time.time()

            train_losses = []
            train_accs = []
            valid_accs = []
            for stage in ["train", "valid"]:
                if stage == "train":
                    model.train()

                else:
                    model.eval()

                loader = dataloaders[stage]

                for batch, target in loader:
                    batch, target = Variable(batch), Variable(target)

                    if cuda:
                        batch, target = batch.cuda(), target.cuda()

                    optimizer.zero_grad()

                    output = model(batch)

                    if stage == "train":
                        # Multiply the correlation by -1 since we want to maximize correlation.
                        loss = -1 * pearson_correlation(output.squeeze(), target.squeeze())
                        loss.backward()
                        optimizer.step()

                        train_losses.append(loss.item())
                        train_accs.append(pearson_correlation(output.squeeze(), target.squeeze()).item())

                    if stage == "valid":
                        valid_accs.append(pearson_correlation(output.squeeze(), target.squeeze()).item())

            print("Epoch {}, train loss {:0.4f}, train acc {:0.4f}, valid acc {:0.4f}, {:0.4f}s"
                  .format(epoch, np.mean(train_losses), np.mean(train_accs),
                          np.mean(valid_accs), time.time() - start_time))

    except KeyboardInterrupt:
        print("Caught keyboard interrupt, testing model.")

    model.eval()

    accs = []
    for batch, target in dataloaders["test"]:
        if cuda:
            batch, target = batch.cuda(), target.cuda()

        output = model(batch)
        accs.append(pearson_correlation(output.squeeze(), target.squeeze()).item())

    print("*" * 40)
    print("Test acc {:0.4f}".format(np.mean(accs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DL Genomics")

    parser.add_argument("--in_channels", type=int, default=3, help="input channels of the genotype data")
    parser.add_argument("--out_channels", type=int, default=16)
    parser.add_argument("--filter_length", type=int, default=26)
    parser.add_argument("--pool_length", type=int, default=3)
    parser.add_argument("--pool_stride", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lstm", type=str2bool, default='true')
    parser.add_argument("--extractor", type=str, default="sae")
    parser.add_argument("--stacks", type=int, default=1)
    parser.add_argument("--intermediate_size", type=int, default=512)

    args = parser.parse_args()

    main(args)
