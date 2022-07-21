import torch
import numpy as np
from cgan import Generator
import json
import pandas as pd

def generate_output_data(noise=None, labels=None):
    model_path = "/ssd003/projects/aieng/public/SyntheticBootcampDatasets/resized_mnist/default_exp/checkpoint_1800.pth"

    checkpoint = torch.load(model_path)

    print(checkpoint.keys())
    gen = Generator(ngf=128, labels_dim=10, features_dim=14*14, latent_dim=100)
    gen.load_state_dict(checkpoint['generator'])
    if noise is None and labels is None:
        noise = torch.tensor(np.random.rand(10, 100))
        label = np.array([0,1,2,3,4,5,6,7,8,9])
        print(noise, label)

    fixed_labels = torch.from_numpy(label).to("cpu")
    fake_features = gen.forward(noise, fixed_labels).detach()
    fake_features = fake_features.cpu().numpy()
    print("output shape ", fake_features.shape)
    print(pd.Series(fake_features.flatten()).to_json(orient='values'))#    print(json.dumps(fake_features, cls=NumpyEncoder))
    return fake_features

if __name__=="__main__":
    generate_output_data()
#def data(self) -> Tuple[np.ndarray, np.ndarray]:
#    """generates fake dataset using the DP-CGAN generator
#
#    Returns:
#        A tuple of generted features and their corresponding labels
#    """
#    fixed_labels = torch.from_numpy(self.fixed_labels).to(self.device)
#    fake_features = self.generator(self.fixed_noise, fixed_labels).detach()

