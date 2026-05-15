import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import UNetColorization, lab_to_rgb  # import your helper
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = UNetColorization()
model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
model.to(device)
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --- Extract frames from video ---
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, frame = vidcap.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{output_folder}/frame_{count:04d}.png", gray)
        count += 1
    vidcap.release()
    print(f"Extracted {count} frames.")
    return count

# --- Colorize a single frame ---
def colorize_frame(image_path, output_path):
    img = Image.open(image_path).convert("L")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_ab = model(input_tensor)

    # Combine L + ab → RGB using lab_to_rgb
    rgb_image = lab_to_rgb(input_tensor[0], output_ab[0])

    # Convert to uint8 for saving
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, rgb_bgr)

# --- Create video from colorized frames ---
def create_video_from_frames(frames_folder, output_video, fps=30):
    frames = sorted(os.listdir(frames_folder))
    frame_example = cv2.imread(os.path.join(frames_folder, frames[0]))
    height, width, layers = frame_example.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame in tqdm(frames, desc="Building video"):
        img = cv2.imread(os.path.join(frames_folder, frame))
        video.write(img)
    video.release()
    print("Video created successfully!")

# --- Full pipeline for a video ---
def process_video(video_path):
    bw_frames_folder = "bw_frames"
    color_frames_folder = "colored_frames"
    output_video_path = "colorized_output.mp4"

    extract_frames(video_path, bw_frames_folder)
    os.makedirs(color_frames_folder, exist_ok=True)

    for filename in tqdm(sorted(os.listdir(bw_frames_folder)), desc="Colorizing frames"):
        colorize_frame(
            os.path.join(bw_frames_folder, filename),
            os.path.join(color_frames_folder, filename)
        )

    create_video_from_frames(color_frames_folder, output_video_path, fps=30)
    return output_video_path
