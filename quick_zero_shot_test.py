#!/usr/bin/env python3
"""
Quick Zero-Shot OpenVLA Test - Just 3 samples for fast verification
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# Set environment
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def quick_zero_shot_test():
    """Very quick zero-shot test - just 3 samples"""
    print("⚡ Quick Zero-Shot OpenVLA Test (3 samples)")
    print("=" * 45)
    
    try:
        # Load model
        print("🔄 Loading OpenVLA...")
        processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Use CPU for stability
        device = "cpu"
        model = model.to(device)
        print("✅ Model loaded on CPU")
        
        # Find one test image
        data_dir = Path("data/scripted_raw")
        if not data_dir.exists():
            print("❌ No data directory found")
            return
        
        # Find first trajectory with images
        test_image = None
        for root, dirs, files in os.walk(data_dir):
            if 'policy_out.pkl' in files:
                img_dir = Path(root) / "images0"
                if img_dir.exists():
                    img_files = list(img_dir.glob("im_*.jpg"))
                    if img_files:
                        test_image = img_files[0]
                        break
        
        if not test_image:
            print("❌ No test image found")
            return
        
        print(f"📸 Using test image: {test_image.name}")
        
        # Test predictions
        instruction = "pick up the object and place it in the bowl"
        print(f"🤖 Instruction: {instruction}")
        
        # Run 3 predictions
        for i in range(3):
            start_time = time.time()
            
            image = Image.open(test_image).convert("RGB")
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            inputs = processor(prompt, image).to(device, dtype=torch.float32)
            
            with torch.inference_mode():
                action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            if hasattr(action, "cpu"):
                action_np = action.cpu().numpy()
            else:
                action_np = np.array(action)
            
            pred_time = time.time() - start_time
            
            print(f"\nPrediction {i+1}:")
            print(f"  Action: [{', '.join([f'{x:.4f}' for x in action_np])}]")
            print(f"  Time: {pred_time:.2f} seconds")
        
        print(f"\n✅ Quick test complete!")
        print(f"💡 For full evaluation, run: python zero_shot_openvla_eval.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_zero_shot_test()
