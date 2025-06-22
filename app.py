import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from vae_model import VAE

st.title("Handwritten Digit Generator (MNIST)")

digit = st.selectbox("Choose a digit to generate", list(range(10)))
device = torch.device("cpu")

model = VAE()
model.load_state_dict(torch.load("vae_mnist.pt", map_location=device))
model.eval()

with torch.no_grad():
    fig, ax = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        z = torch.randn(1, 20)
        digit_tensor = F.one_hot(torch.tensor([digit]), num_classes=10).float()
        z_cond = torch.cat((z, digit_tensor), dim=1)
        sample = model.decode(z_cond).view(28, 28).numpy()
        ax[i].imshow(sample, cmap="gray")
        ax[i].axis('off')
    st.pyplot(fig)
