import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np


# -----------------------------------------
# 1. Model Definition (Must match training!)
# -----------------------------------------
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# -----------------------------------------
# 2. Helper Functions
# -----------------------------------------
@st.cache_resource
def load_model():
    """
    Loads the model and weights.
    Cached so we don't reload on every interaction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet().to(device)

    try:
        # Load weights
        model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        return None, device


def process_image(image):
    """
    Preprocesses the user image to match MNIST format:
    1. Grayscale
    2. Resize to 28x28
    3. Normalize to [-1, 1]
    """
    # Convert to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Transform to Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Add batch dimension (1, 1, 28, 28)
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor


# -----------------------------------------
# 3. Streamlit UI
# -----------------------------------------
st.title("🔢 MNIST Digit Classifier")
st.write("Upload an image of a digit (0-9) to check the prediction.")

# Load Model
model, device = load_model()

if model is None:
    st.error("Model file 'mnist_model.pth' not found! Please run 'mnist_classifier.py' first.")
else:
    # Sidebar options
    st.sidebar.header("Options")
    invert_colors = st.sidebar.checkbox("Invert Colors (White Background?)", value=True)

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)

        # MNIST images are black background with white digits.
        # Real world photos are often white background with black digits.
        # We might need to invert them.
        if invert_colors:
            image = ImageOps.invert(image.convert("RGB"))

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption='Processed Image Input', width=150)

        # Inference
        img_tensor = process_image(image).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred_class = torch.max(probs, 1)

        # Display Result
        with col2:
            st.metric(label="Predicted Digit", value=str(pred_class.item()))
            st.write(f"Confidence: **{conf.item() * 100:.2f}%**")

        # Bar chart of probabilities
        st.subheader("Class Probabilities")
        probs_np = probs.cpu().numpy().flatten()
        st.bar_chart(probs_np)