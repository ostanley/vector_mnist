import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
#from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from trainer import Trainer

from cgan import Discriminator, Generator  # isort:skip

from dataset import class_ratios, split_dataset, TabularDataset  # isort:skip

from utils import init_weights, setup_logging  # isort:skip

logger = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=logger)

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    """Setup arguments and training hyperparameters"""

    parser = argparse.ArgumentParser(description="DP-CGAN params")
    parser.add_argument("--seed", type=int, default=42069)

    # Opacus DP args
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. \
            Comes at a performance cost",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.55,
        help="Target epsilon",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.15,
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Target delta (default: 1e-5)",
    )

    # CGAN args
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=100,
        help="Size of the latent z vector",
    )
    parser.add_argument(
        "--ngf",
        type=int,
        default=128,
        help="Number of features in the generator layers",
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=128,
        help="Number of features in the discriminator layers",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to spend the privacy budget for",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate, default=0.0002",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="beta2 for adam. default=0.999",
    )

    # MLP evaluation args
    parser.add_argument(
        "--eval-period",
        type=int,
        default=100,
        help="Evaluation and checkpointing interval",
    )

    # Dataset args
    # Need to add our data here 
    parser.add_argument(
        "--dset-src",
        type=str,
        default="/ssd003/projects/aieng/public/SyntheticBootcampDatasets/resized_mnist/data/"
        "data.pkl",
    )
    parser.add_argument(
        "--dset-name",
        type=str,
        default="resized_mnist",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="training batch size"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.2,
        help="test ratio for splitting the dataset",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="number of data loading workers"
    )
    parser.add_argument(
        "--synth-data-size",
        type=int,
        default=250000,
        help="The size of synthetic dataset for evaluation",
    )

    # Model args
    parser.add_argument(
        "--checkpoint", type=str, default="", help="path to checkpoint file"
    )

    parser.add_argument("--exp-dir", type=str, default="./100_epochs")

    args = parser.parse_args()
    return args


def main(args):

    os.makedirs(args.exp_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device ="cuda" if torch.cuda.is_available() else "cpu"

    (train_features, train_labels, test_features, test_labels) = split_dataset(
        args.split_ratio, args.dset_name, args.dset_src
    )

    train_dataset = TabularDataset(train_features, train_labels)
    test_dataset = TabularDataset(test_features, test_labels)

    features_dim = train_dataset.num_features()
    labels_dim = train_dataset.labels_dim()

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True,
    )

    generator = Generator(
        latent_dim=args.latent_dim,
        ngf=args.ngf,
        features_dim=features_dim,
        labels_dim=labels_dim,
    )
    generator = generator.to(device)
    generator.apply(init_weights)

    optimizerG = optim.Adam(
        generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )

    discriminator = Discriminator(
        ndf=args.ndf, features_dim=features_dim, labels_dim=labels_dim
    )
    discriminator = discriminator.to(device)
    discriminator.apply(init_weights)

    optimizerD = optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )

    #privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)

    if args.checkpoint != "":
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizerG.load_state_dict(checkpoint["optimizerG"])
        optimizerD.load_state_dict(checkpoint["optimmizerD"])
     #   privacy_engine.accountant = checkpoint["accountant"]

        logger.info(f"checkpoint loaded from {args.checkpoint}")

    # Opacus adds privacy-related responsibilites to the main PyTorch
    # training objects: model, optimizer, and the data loader.
    #(
    #    discriminator,
    #    optimizerD,
    #    train_loader,
    #) = privacy_engine.make_private_with_epsilon(
     #   module=discriminator,
     #   optimizer=optimizerD,
     #   data_loader=train_loader,
     #  max_grad_norm=args.max_per_sample_grad_norm,
      #  target_epsilon=args.epsilon,
      # target_delta=args.delta,
      #  epochs=args.epochs,
    #)

    criterion = nn.BCELoss()

    fakedata_size = min(train_dataset.__len__(), args.synth_data_size)
    fixed_noise = torch.randn(fakedata_size, args.latent_dim, device=device)
    classes, ratios = class_ratios(train_labels)
    fixed_labels = rng.choice(classes, size=fakedata_size, p=ratios)

    trainer = Trainer(
        generator=generator,
        optimizerG=optimizerG,
        discriminator=discriminator,
        optimizerD=optimizerD,
        criterion=criterion,
     #   privacy_engine=privacy_engine,
        train_loader=train_loader,
        test_dataset=test_dataset,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        device=device,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        fixed_noise=fixed_noise,
        fixed_labels=fixed_labels,
        exp_dir=args.exp_dir,
        eval_period=args.eval_period,
    )

    mlp_aucs, mlp_aps = trainer.train()

    iters = np.arange(
        args.eval_period,
        (len(mlp_aucs) + 1) * args.eval_period,
        args.eval_period,
    )
    df = pd.DataFrame(
        {
            "iter": iters,
            "mlp_auc": np.array(mlp_aucs),
            "mlp_ap": np.array(mlp_aps),
        }
    )
    df.plot(
        x="iter",
        y=["mlp_auc", "mlp_ap"],
        kind="line",
        xticks=iters // args.eval_period,
        xlabel=f"iters/{args.eval_period}",
    )
    plt.savefig(f"{args.exp_dir}/mlp_auc.png")
    df.to_csv(f"{args.exp_dir}/mlp_metrics.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
