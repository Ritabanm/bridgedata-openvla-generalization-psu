# scripts/quick_inference_test.py
# Fixed OpenVLA-7B inference test for Mac MPS (float16 dtype, no unnorm_key in generate)

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests
from io import BytesIO

print("=== OpenVLA Inference Test Starting ===")

# Device & dtype (float16 for MPS compatibility)
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
print(f"Device: {device} | Data type: {dtype}")

# Load processor and model
print("Loading processor and model... (cached if already downloaded)")
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    attn_implementation="eager"
).to(device)

print("Model loaded successfully!")

# Load sample image (use a local file you downloaded)
print("Loading local test image...")
image_path = "test_robot.jpg"  # CHANGE TO YOUR ACTUAL SAVED FILENAME
try:
    image = Image.open(image_path).convert("RGB")
    print("Image loaded successfully!")
except Exception as e:
    print(f"Image load failed: {e}")
    print("Please download a robot arm image and save as 'test_robot.jpg' in scripts/")
    exit(1)

# Test prompt
prompt = "In: What action should the robot take to pick up the red block and put it in the bowl?\nOut:"

# Process inputs
print("Processing prompt and image...")
inputs = processor(prompt, image).to(model.device, dtype=dtype)

# Run inference (no unnorm_key here - use raw tokens for now)
print("Generating action tokens (10–60 seconds)...")
with torch.no_grad():
    action_tokens = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )

# Results
decoded = processor.decode(action_tokens[0], skip_special_tokens=True)
print("\n=== RESULTS ===")
print("Raw action tokens:", action_tokens.tolist())
print("Decoded output:", decoded)
print("=== Test Complete! ===")

# Note: For real BridgeData actions, use model.predict_action(...) after generation