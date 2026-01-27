#!/usr/bin/env python3
"""
Fast OpenVLA Test with Progress - Shows what's happening during slow inference
"""

import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Force CPU for stability
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def fast_openvla_test():
    """Fast test with progress indicators"""
    print("⚡ Fast OpenVLA Test with Progress")
    print("=" * 40)
    
    try:
        # Load model
        print("🔄 Loading OpenVLA model...")
        start_time = time.time()
        
        processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to("cpu")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        # Find test image
        data_dir = Path("data/scripted_raw")
        test_image = None
        
        print("🔍 Finding test image...")
        for root, dirs, files in os.walk(data_dir):
            if 'policy_out.pkl' in files:
                img_dir = Path(root) / "images0"
                if img_dir.exists():
                    img_files = list(img_dir.glob("im_*.jpg"))
                    if img_files:
                        test_image = img_files[0]
                        print(f"📸 Found: {test_image.name}")
                        break
        
        if not test_image:
            print("❌ No test image found")
            return
        
        # Test with progress
        instruction = "pick up the object"
        print(f"\n🤖 Testing with instruction: '{instruction}'")
        print("⏳ Processing image (this takes 10-30 seconds)...")
        
        # Step 1: Load image
        print("   📷 Loading image...")
        step_start = time.time()
        image = Image.open(test_image).convert("RGB")
        print(f"   ✅ Image loaded in {time.time() - step_start:.1f}s")
        
        # Step 2: Process with processor
        print("   🔄 Processing with tokenizer...")
        step_start = time.time()
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, image).to("cpu", dtype=torch.float32)
        print(f"   ✅ Processed in {time.time() - step_start:.1f}s")
        
        # Step 3: Model inference (the slow part)
        print("   🧠 Running model inference (this is the slow part)...")
        step_start = time.time()
        
        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        
        inference_time = time.time() - step_start
        print(f"   ✅ Inference completed in {inference_time:.1f}s")
        
        # Step 4: Convert result
        print("   📊 Converting result...")
        step_start = time.time()
        
        if hasattr(action, "cpu"):
            action_np = action.cpu().numpy()
        else:
            action_np = torch.tensor(action).numpy()
        
        print(f"   ✅ Converted in {time.time() - step_start:.1f}s")
        
        # Show results
        total_time = load_time + inference_time
        print(f"\n🎯 RESULTS:")
        print(f"   Action (7-DoF): [{', '.join([f'{x:.4f}' for x in action_np])}]")
        print(f"   Model loading: {load_time:.1f}s")
        print(f"   Inference time: {inference_time:.1f}s")
        print(f"   Total time: {total_time:.1f}s")
        
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        print(f"   - First prediction is slowest: {inference_time:.1f}s")
        print(f"   - Subsequent predictions are usually faster")
        print(f"   - CPU inference: ~{inference_time:.0f}s per prediction")
        print(f"   - For 9 samples: ~{9 * inference_time:.0f}s total")
        
        print(f"\n✅ OpenVLA zero-shot test successful!")
        
        return action_np, inference_time
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def show_timing_estimate(inference_time):
    """Show timing estimates for different sample sizes"""
    print(f"\n⏱️  TIMING ESTIMATES (based on {inference_time:.1f}s per inference):")
    print(f"   1 sample:  {inference_time:.0f}s")
    print(f"   3 samples: {3 * inference_time:.0f}s") 
    print(f"   9 samples: {9 * inference_time:.0f}s")
    print(f"   27 samples: {27 * inference_time:.0f}s")
    print(f"\n💡 Recommendation: Use 3-9 samples for quick tests")

if __name__ == "__main__":
    action, inference_time = fast_openvla_test()
    if action is not None:
        show_timing_estimate(inference_time)
