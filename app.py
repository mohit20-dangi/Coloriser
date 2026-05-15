import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn
import tempfile
from model import UNetColorization, lab_to_rgb  # Import model and conversion function
from video import process_video  # Import video functions

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = UNetColorization()
model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
model.to(device)
model.eval()

# --- Transform for images ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --- Streamlit UI ---
st.set_page_config(page_title="Image & Video Colorization", layout="centered")
st.title("🎨 Image & Video Colorization App")
st.write("Upload a **black & white image** or **video**, and watch it come alive in color!")

# Choose input type
option = st.radio("Select input type:", ("Image", "Video"))

# ------------------------------------------------------
# IMAGE COLORIZATION SECTION
# ------------------------------------------------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded B/W Image", use_container_width=True)

        if st.button("🪄 Colorize Image"):
            with st.spinner("Colorizing... please wait"):
                # Prepare input tensor
                input_L = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output_ab = model(input_L)  # output shape: [1, 2, H, W]

                # Convert LAB → RGB using the helper function from model.py
                rgb_image = lab_to_rgb(input_L[0], output_ab[0])

                st.image(rgb_image, caption="Colorized Image", use_container_width=True)
                st.success("✅ Image colorization complete!")

# ------------------------------------------------------
# VIDEO COLORIZATION SECTION
# ------------------------------------------------------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a grayscale video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            tmp_video_path = tmp_video.name

        with open(tmp_video_path, "rb") as f:
            video_bytes = f.read()
        st.video(tmp_video_path)
        if st.button("🎬 Colorize Video"):
            with st.spinner("Processing video... this may take a while"):
                output_video_path = process_video(tmp_video_path)
                st.video(output_video_path)
                # --- Download button for colorized video ---
                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()

                st.download_button(
                    label="⬇️ Download Colorized Video",
                    data=video_bytes,
                    file_name="colorized_video.mp4",
                    mime="video/mp4"
                )
                st.success("✅ Video colorization complete!")
