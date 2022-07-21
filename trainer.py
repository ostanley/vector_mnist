import logging
from dataclasses import dataclass
from typing import Tuple

import cgan
import dataset
import numpy as np
import opacus
import torch
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=logger)


@dataclass
class Trainer:
    """A trainer for CGAN with differential privacy"""

    generator: cgan.Generator
    optimizerG: torch.optim
    discriminator: cgan.Discriminator
    optimizerD: torch.optim
    criterion: torch.nn.modules.loss
    #privacy_engine: opacus.privacy_engine.PrivacyEngine
    train_loader: torch.utils.data.dataloader.DataLoader
    test_dataset: dataset.TabularDataset
    target_epsilon: float
    target_delta: float
    device: str
    epochs: int
    latent_dim: int
    fixed_noise: torch.Tensor
    fixed_labels: np.ndarray
    exp_dir: str
    eval_period: int

    def __post_init__(self):
        self.n_classes = self.test_dataset.num_classes()

    def train(self) -> Tuple[list, list]:
        """Trains the CGAN model with differetial privacy

        Returns:
            A tuple containing list of evaluation metrics: AUC and average
            precision
        """

        iteration = 0
        best_auc = 0
        budget = True
        mlp_aucs = []
        mlp_aps = []

        for epoch in range(self.epochs):

            #if not budget:
            #    logger.info(
            #        "Privacy budget exceeded! Iteration = %d, ε = %.3f"
            #        % (iteration, epsilon)  # noqa: F821
            #    )
            #    break

            for _, data in enumerate(self.train_loader):

                real_data = data[0].type(torch.FloatTensor).to(self.device)
                real_labels = data[1].to(self.device)
                batch_size = real_data.size(0)

                label_fake = torch.full(
                    (batch_size, 1), 0.0, device=self.device
                )
                label_true = torch.full(
                    (batch_size, 1), 1.0, device=self.device
                )

                ######################
                # (1) Update D network
                ######################
                self.optimizerD.zero_grad()

                # train with fake data
                noise = torch.randn(
                    batch_size, self.latent_dim, device=self.device
                )
                gen_labels = torch.randint(
                    0, self.n_classes, (batch_size,), device=self.device
                )

                fake = self.generator(noise, gen_labels)

                output = self.discriminator(fake.detach(), gen_labels)
                errD_fake = self.criterion(output, label_fake)
                errD_fake.backward()
                self.optimizerD.step()
                self.optimizerD.zero_grad()

                # train with real data
                output = self.discriminator(real_data, real_labels)
                errD_real = self.criterion(output, label_true)
                errD_real.backward()
                self.optimizerD.step()

                errD = errD_real + errD_fake

                ######################
                # (2) Update G network
                ######################
                self.optimizerG.zero_grad()
                self.optimizerD.zero_grad()

                output_g = self.discriminator(fake, gen_labels)
                errG = self.criterion(output_g, label_true)
                errG.backward()

                self.optimizerG.step()

                #
                #(
                #    epsilon,
                #    best_alpha,
                #) = self.privacy_engine.accountant.get_privacy_spent(
                #    delta=self.target_delta
                #)

                #if epsilon > self.target_epsilon:
                #    budget = False
                #    break

                iteration = iteration + 1
                if iteration % self.eval_period == 0:

                    logger.info(
                        "Iteration = %d, Loss_D = %.2f, Loss_G = %.2f"
                        % (iteration, errD.item(), errG.item())
                    )
                    #logger.info(
                    #    "(ε = %.3f, δ = %.2f) for α = %.2f"
                    #    % (epsilon, self.target_delta, best_alpha)
                    #)
                    mlp_auc, mlp_ap = self._eval()
                    mlp_aucs.append(mlp_auc)
                    mlp_aps.append(mlp_ap)
                    logger.info(
                        "mlp_auc = %.3f, mlp_ap = %.3f" % (mlp_auc, mlp_ap)
                    )
                    if mlp_auc > best_auc:
                        best_auc = mlp_auc
                        logger.info(
                            f"Checkpoint saved at iteration={iteration} "
                           # f"eps={epsilon:.3f}"
                        )
                        torch.save(
                            {
                                "discriminator": (
                                    self.discriminator.state_dict()
                                ),
                                "generator": self.generator.state_dict(),
                                #"accountant": self.privacy_engine.accountant,
                                "optimizerG": self.optimizerG.state_dict(),
                                "optimmizerD": self.optimizerD.state_dict(),
                            },
                            f"{self.exp_dir}/checkpoint_{iteration}.pth"
                            #f"{epsilon:.3f}.pth",
                        )

        return mlp_aucs, mlp_aps

    def _eval(self) -> Tuple[float, float]:
        """Evaluates the model by applying mlp classifier

        Returns:
            A tuple containing the classifier AUC and average precision
        """
        fake_features, fake_labels = self._generate_fake_data()
        mlp = MLPClassifier(early_stopping=True).fit(
            fake_features, fake_labels
        )
        class_probs = mlp.predict_proba(self.test_dataset.features)
        auc = metrics.roc_auc_score(
            self.test_dataset.labels,
            class_probs,
            average="weighted",
            multi_class="ovo",
        )

        if self.n_classes > 2:
            lb = LabelBinarizer()
            lb.fit(self.test_dataset.labels)
            y_test = lb.transform(self.test_dataset.labels)
        else:
            y_test = self.test_dataset.labels

        ap = metrics.average_precision_score(y_test, class_probs)
        return auc, ap

    def _generate_fake_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """generates fake dataset using the DP-CGAN generator

        Returns:
            A tuple of generted features and their corresponding labels
        """
        fixed_labels = torch.from_numpy(self.fixed_labels).to(self.device)
        fake_features = self.generator(self.fixed_noise, fixed_labels).detach()
        fake_features = fake_features.cpu().numpy()
        return fake_features, self.fixed_labels
