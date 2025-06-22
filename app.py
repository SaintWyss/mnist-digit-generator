import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from vae_model import VAE  # asegurate de que esté en el mismo folder

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
        sample = model.decode(z).view(28, 28).numpy()
        ax[i].imshow(sample, cmap="gray")
        ax[i].axis('off')
    st.pyplot(fig)