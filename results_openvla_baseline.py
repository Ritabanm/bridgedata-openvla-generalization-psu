#!/usr/bin/env python3
"""
Hardcoded OpenVLA predictions from current baseline run
"""

import numpy as np
import argparse

# Hardcoded predictions from your current baseline run
# Replace these with actual values from your terminal output
HARDCODED_PREDICTIONS = [
    # Sample 1, Timestep 1
    {
        'predicted': [-0.0083, 0.0177, 0.0380, 0.0420, -0.0611, 0.0999, 0.0000],
        'ground_truth': [-0.0291, 0.0698, 0.0007, -0.0083, 0.0017, -0.6787, 0.0609],
        'sample': 1, 'timestep': 1
    },
    # Sample 1, Timestep 2  
    {
        'predicted': [-0.0074, 0.0119, 0.0400, 0.0439, -0.0926, 0.0999, 0.0000],
        'ground_truth': [-0.0321, 0.0641, -0.0044, 0.0002, 0.0026, -0.6839, 0.9952],
        'sample': 1, 'timestep': 2
    },
 #Sample 2, Timestep 1 
    {
        'predicted': [-0.0029, -0.0050, -0.0153, 0.0158, 0.0133, -0.0017, 0.9961],
        'ground_truth': [-0.0030, -0.0366, 0.0037, -0.0035, 0.0072, 0.3568, 0.9934],
        'sample': 2, 'timestep': 1
    },

    #Sample 2, Timestep 2

    {
        'predicted': [0.0025, 0.0015, -0.0057, 0.0145, 0.0059, 0.0112, 0.0000],
        'ground_truth': [-0.0021, -0.0359, -0.0042, -0.0047, 0.0044, 0.3398, 0.9956],
        'sample': 2, 'timestep': 2
    },
    # Sample 3, Timestep 1
    {
        'predicted': [-0.0132, -0.0173, -0.0106, 0.0177, 0.0106, 0.0128, 0.9961],
        'ground_truth': [-0.0059, 0.0033, 0.0010, -0.0062, -0.0001, 0.4447, 1.0000],
        'sample': 3, 'timestep': 1
    },
    #Sample 3, Timestep 2
    {
        'predicted': [-0.0022, -0.0011, -0.0156, 0.0037, 0.0441, -0.0001, 0.9961],
        'ground_truth': [-0.0011, 0.0052, -0.0267, -0.0026, 0.0080, 0.4295, 1.0000],
        'sample': 3, 'timestep': 2
    },
    #Sample 4, Timestep 1
    {
        'predicted': [0.0023, 0.0018, -0.0008, 0.0145, 0.0086, 0.0112, 0.0000],
        'ground_truth': [-0.0243, -0.0485, -0.0007, -0.0077, -0.0003, 0.7667, 0.9955],
        'sample':4, 'timestep':1
    },
    
    #Sample 4, Timestep 2
    {
        'predicted': [0.0023, 0.0018, -0.0023, 0.0145, -0.0015, -0.0081, 0.0000],
        'ground_truth': [-0.0284, -0.0369, -0.0021, -0.0025, 0.0000, 0.7629, 0.9945],
        'sample':4, 'timestep':2
    },

    #Sample 5, Timestep 1
    {
        'predicted': [0.0034, 0.0122, 0.0218, 0.0037, -0.0537, 0.0128, 0.0000],
        'ground_truth': [-0.0076, 0.0858, 0.0006, 0.0033, -0.0000, -0.6684, 1.0000],
        'sample':5, 'timestep':1
    },

    #Sample 5, Timestep 2

    {
        'predicted': [-0.0031, 0.0161, 0.0208, -0.0199, -0.0343, 0.0531, 0.0000],
        'ground_truth': [-0.0142, 0.0841, -0.0027, 0.0081, 0.0029, -0.6480, 1.0000],
        'sample':6, 'timestep':2
    },

    #Sample 6, Timestep 1
    {
        'predicted': [0.0108, -0.0014, -0.0153, -0.0180, 0.0186, 0.0112, 0.9961],
        'ground_truth': [0.0243, 0.0290, -0.0028, -0.0030, -0.0083, -0.1676, 0.9938],
        'sample':6, 'timestep':1
    },

    #Sample 6, Timestep 2
    {
        'predicted': [-0.0045, -0.0075, -0.0148, 0.0114, 0.0428, -0.0259, 0.9961],
        'ground_truth': [0.0198, 0.0266, -0.0026, -0.0014, 0.0061, -0.1630, 0.9945],
        'sample':6, 'timestep':2
    },

    #Sample 7, Timestep 1

    {
        'predicted': [-0.0036, -0.0211, -0.0236, -0.0512, 0.0193, -0.0629, 0.0000],
        'ground_truth': [0.0032, -0.0558, -0.0014, 0.0030, 0.0015, 0.6495, 1.0000],
        'sample': 7, 'timestep':1
    },

    #Sample 7, Timestep 2
    {
        'predicted': [0.0029, -0.0231, -0.0213, -0.0327, 0.0153, -0.0629, 0.0000],
        'ground_truth': [0.0169, -0.0490, -0.0041, 0.0063, -0.0045, 0.6170, 0.9950],
        'sample': 7, 'timestep':2
    },

        #Sample 8, Timestep 1
    {
        'predicted': [0.0031, 0.0083, -0.0018, 0.0177, 0.0052, -0.0001, 0.0000],
        'ground_truth': [-0.0387, -0.0348, 0.0032, -0.0012, 0.0002, -0.2183, 0.0596],
        'sample': 8, 'timestep':1
    },
    #Sample 8, Timestep 2
    {
        'predicted': [-0.0051, 0.0187, -0.0010, 0.0011, -0.0048, 0.0289, 0.0000],
        'ground_truth': [-0.0334, -0.0257, 0.0037, -0.0054, 0.0027, -0.2083, 0.0633],
        'sample': 8, 'timestep':2
    },

    #Sample 9, Timestep 1

    {
        'predicted': [-0.0080, -0.0338, -0.0145, -0.0206, 0.0193, -0.0307, 0.0000],
        'ground_truth': [0.0134, 0.0466, 0.0040, -0.0087, -0.0012, 0.7999, 0.0689],
        'sample':9, 'timestep':1
    },
#Sample 9, Timestep 2
    {
        'predicted': [0.0159, -0.0082, -0.0236, 0.0280, 0.0267, -0.0339, 0.9961],
        'ground_truth': [0.0234, 0.0557, 0.0029, -0.0022, 0.0014, 0.7811, 0.0558],
        'sample': 9, 'timestep':2
    },

    #Sample 10, Timestep 1
    {
        'predicted': [0.0000, 0.0090, -0.0034, 0.0644, -0.0068, -0.0001, 0.9961],
        'ground_truth': [-0.0041, -0.0255, -0.0027, 0.0002, -0.0044, -0.7090, 1.0000],
        'sample': 10, 'timestep':1
    },

    #Sample 10, Timestep 2
    {
        'predicted': [-0.0060, -0.0163, 0.0400, -0.0244, -0.0926, -0.0436, 0.9961],
        'ground_truth': [0.0001, -0.0147, -0.0077, 0.0037, -0.0022, -0.6958, 1.0000],
        'sample': 10, 'timestep':2
    },
       #Sample 11, Timestep 1
    {
        'predicted': [-0.0033, -0.0017, -0.0057, -0.0014, 0.0039, -0.0097, 0.9961],
        'ground_truth': [-0.0212, 0.0174, 0.0000, -0.0037, -0.0027, -0.4425, 0.9989],
        'sample': 11, 'timestep':1
    },
    #Sample 11, Timestep 2
    {
        'predicted': [0.0011, -0.0111, -0.0067, 0.0043, -0.0323, -0.0242, 0.9961],
        'ground_truth': [-0.0214, 0.0196, -0.0041, 0.0009, 0.0060, -0.4308, 1.0000],
        'sample': 11, 'timestep':2
    },

    #Sample 12, Timestep 1
    {
        'predicted': [-0.0004, -0.0001, 0.0013, -0.0021, -0.0028, -0.0001, 0.9961],
        'ground_truth': [-0.0163, -0.0294, 0.0012, -0.0030, 0.0026, -0.4423, 0.9934],
        'sample': 12, 'timestep':1
    },

    #Sample 12, Timestep 2
    {
        'predicted': [-0.0036, 0.0336, -0.0080, -0.0238, 0.0227, 0.1563, 0.0000],
        'ground_truth': [-0.0100, -0.0322, -0.0096, -0.0032, -0.0051, -0.4234, 0.9953],
        'sample': 12, 'timestep':2
    },

    #Sample 13, Timestep 1
    {
        'predicted': [-0.0069, -0.0079, 0.0044, -0.0219, -0.0504, -0.1000, 0.9961],
        'ground_truth': [0.0013, -0.0053, -0.0347, -0.0026, -0.0006, 0.0974, 1.0000],
        'sample': 13, 'timestep':1
    },

    #Sample 13, Timestep 2
    {
        'predicted': [-0.0074, -0.0208, 0.0044, -0.0219, -0.0450, -0.1097, 0.9961],
        'ground_truth': [0.0108, -0.0034, -0.0221, -0.0065, 0.0049, 0.0774, 0.9996],
        'sample': 13, 'timestep':2
    },

    #Sample 14, Timestep 1

    {
        'predicted': [-0.0069, 0.0148, -0.0052, 0.0318, 0.0140, 0.0096, 0.0000],
        'ground_truth': [0.0181, -0.0346, -0.0046, -0.0038, -0.0064, 0.7927, 1.0000],
        'sample': 14, 'timestep':1
    },

    #Sample 14, Timestep 2
    
    {
        'predicted': [-0.0038, 0.0041, -0.0065, 0.0561, -0.0215, -0.0130, 0.0000],
        'ground_truth': [0.0054, -0.0458, -0.0062, 0.0075, 0.0009, 0.7680, 0.9958],
        'sample': 14, 'timestep':2
    },

    #Sample 15, Timestep 1
    {
        'predicted': [0.0000, -0.0043, -0.0002, 0.0209, -0.0015, -0.0081, 0.0000],
        'ground_truth': [0.0194, 0.0260, 0.0015, -0.0058, -0.0044, 0.7748, 0.0763],
        'sample': 15, 'timestep':1
    },

    #Sample 15, Timestep 2
    {
        'predicted': [-0.0062, -0.0069, -0.0044, -0.0257, -0.0162, -0.0017, 0.9961],
        'ground_truth': [0.0226, 0.0173, 0.0014, -0.0012, -0.0004, 0.7697, 0.0687],
        'sample': 15, 'timestep':2
    },

    #Sample 16, Timestep 1
    {
        'predicted': [-0.0152, -0.0046, -0.0054, 0.0126, -0.0061, 0.0870, 0.9961],
        'ground_truth': [0.0169, -0.0110, 0.0022, -0.0002, 0.0112, 0.5428, 1.0000],
        'sample': 16, 'timestep':1
    },

        #Sample 16, Timestep 2
    {
        'predicted': [-0.0051, -0.0169, -0.0036, 0.0126, 0.0052, -0.0178, 0.9961],
        'ground_truth': [0.0208, -0.0022, -0.0023, 0.0055, 0.0086, 0.5372, 0.9981],
        'sample': 16, 'timestep':2
    },

    #Sample 17, Timestep 1
    
    {
        'predicted': [-0.0007, 0.0167, 0.0013, -0.0372, -0.0088, 0.0515, 0.0000],
        'ground_truth': [-0.0072, -0.0069, 0.0015, -0.0004, -0.0029, 0.7976, 1.0000],
        'sample': 17, 'timestep':1
    },

    #Sample 17, Timestep 2
    {
        'predicted': [0.0213, 0.0252, -0.0073, 0.0618, 0.0454, 0.0322, 0.0000],
        'ground_truth': [-0.0165, -0.0003, 0.0060, -0.0054, 0.0053, 0.7756, 1.0000],
        'sample': 17, 'timestep':2
    },

    #Sample 18, Timestep 1
    {
        'predicted': [0.0031, 0.0329, 0.0057, -0.0097, -0.0416, 0.0692, 0.0000],
        'ground_truth': [-0.0151, 0.0128, 0.0028, 0.0056, -0.0052, 0.2289, 0.9913],
        'sample': 18, 'timestep':1
    },

    #Sample 18, Timestep 2
    {
        'predicted': [-0.0009, 0.0018, -0.0148, 0.0171, 0.0655, 0.0322, 0.0000],
        'ground_truth': [-0.0122, 0.0059, 0.0033, 0.0006, -0.0028, 0.2130, 0.9980],
        'sample': 18, 'timestep':2
    },

    #Sample 19, Timestep 1
    {
        'predicted': [0.0085, 0.0012, -0.0236, 0.0650, 0.0267, 0.0418, 0.9961],
        'ground_truth': [0.0165, -0.0370, -0.0010, -0.0001, 0.0057, -0.4236, 1.0000],
        'sample': 19, 'timestep':1
    },

    #Sample 19, Timestep 2
    
    {
        'predicted': [0.0027, 0.0009, -0.0034, 0.0101, -0.0048, -0.0242, 0.9961],
        'ground_truth': [0.0108, -0.0344, -0.0037, -0.0016, -0.0046, -0.4194, 1.0000],
        'sample': 19, 'timestep':2
    },

    #Sample 20, Timestep 1
    {
        'predicted': [-0.0033, 0.0187, -0.0002, -0.0321, -0.0162, 0.0499, 0.9961],
        'ground_truth': [0.0209, -0.0393, 0.0051, 0.0034, -0.0015, -0.6995, 1.0000],
        'sample': 20, 'timestep':1
    },

    #Sample 20, Timestep 2
    {
        'predicted': [-0.0033, -0.0156, 0.0086, -0.0225, -0.0356, 0.0354, 0.9961],
        'ground_truth': [0.0208, -0.0448, 0.0007, 0.0074, -0.0078, -0.6986, 0.9884],
        'sample': 20, 'timestep':2
    },
    
    #Sample 21, Timestep 1
    {
        'predicted': [-0.0049, -0.0017, -0.0028, 0.0050, 0.0106, 0.0451, 0.9961],
        'ground_truth': [0.0000, -0.0323, 0.0042, -0.0015, 0.0080, 0.3114, 1.0000],
        'sample': 21, 'timestep':1
    },
    
    #Sample 21, Timestep 2
    {
        'predicted': [0.0146, 0.0261, -0.0075, -0.0065, 0.0032, 0.0950, 0.0000],
        'ground_truth': [-0.0034, -0.0303, -0.0047, 0.0062, -0.0004, 0.2925, 0.9967],
        'sample': 21, 'timestep':2
    },

    #Sample 22, Timestep 1
    {
        'predicted': [0.0000, -0.0082, 0.0042, -0.0282, -0.0390, -0.0452, 0.0000],
        'ground_truth': [0.0345, -0.0424, -0.0044, -0.0018, 0.0048, 0.9367, 0.0518],
        'sample': 22, 'timestep':1
    },

    #Sample 22, Timestep 2
    {
        'predicted': [-0.0022, -0.0004, 0.0008, -0.0014, 0.0173, -0.0130, 0.9961],
        'ground_truth': [0.0184, -0.0279, 0.0024, -0.0018, -0.0002, 0.9139, 0.0575],
        'sample': 22, 'timestep':2
    },
    
    #Sample 23, Timestep 1
    {
        'predicted': [-0.0058, -0.0046, 0.0055, -0.0397, -0.0162, -0.0210, 0.0000],
        'ground_truth': [0.0021, -0.0291, 0.0024, 0.0071, 0.0069, -0.9734, 0.9993],
        'sample': 23, 'timestep':1
    },
    
    #Sample 23, Timestep 2
    {
        'predicted': [-0.0002, -0.0105, 0.0187, -0.0187, -0.0524, -0.0162, 0.9961],
        'ground_truth': [0.0018, -0.0231, 0.0037, 0.0017, -0.0037, -0.9650, 0.9939],
        'sample': 23, 'timestep':2
    },

    #Sample 24, Timestep 1
    {
        'predicted': [0.0081, 0.0015, -0.0075, -0.0206, 0.0059, 0.0209, 0.0000],
        'ground_truth': [-0.0170, -0.0371, -0.0083, 0.0035, -0.0063, -0.6241, 0.0547],
        'sample': 24, 'timestep':1
    },

    #Sample 24, Timestep 2
    {
        'predicted': [0.0081, -0.0134, -0.0021, -0.0116, -0.0175, -0.0323, 0.0000],
        'ground_truth': [-0.0171, -0.0401, 0.0019, 0.0021, -0.0049, -0.6082, 0.0558],
        'sample': 24, 'timestep':2
    },

    #Sample 25, Timestep 1

    {
        'predicted': [-0.0154, 0.0141, 0.0039, -0.0161, -0.0135, 0.0241, 0.9961],
        'ground_truth': [0.0058, 0.0124, 0.0002, -0.0092, 0.0055, -0.1390, 0.9988],
        'sample': 25, 'timestep':1
    },

    #Sample 25, Timestep 2
    {
        'predicted': [-0.0154, 0.0141, -0.0073, 0.0101, -0.0108, 0.1579, 0.9961],
        'ground_truth': [0.0013, 0.0134, 0.0012, -0.0005, -0.0120, -0.1349, 0.9912],
        'sample': 25, 'timestep':2
    },

    #Sample 26, Timestep 1
   {
        'predicted': [-0.0116, 0.0381, 0.0112, 0.0056, -0.0249, 0.1563, 0.0000],
        'ground_truth': [0.0346, 0.0529, -0.0023, -0.0049, -0.0076, 0.7310, 0.0699],
        'sample': 26, 'timestep':1
    },

    #Sample 26, Timestep 2
   {
        'predicted': [-0.0036, 0.0219, 0.0018, 0.0158, -0.0108, 0.0289, 0.0000],
        'ground_truth': [0.0420, 0.0557, 0.0045, 0.0021, 0.0035, 0.7388, 0.0592],
        'sample': 26, 'timestep':2
    },

        #Sample 27, Timestep 1
   {
        'predicted': [0.0072, 0.0106, -0.0013, 0.0094, -0.0068, 0.0161, 0.0000], 
        'ground_truth': [-0.0082, 0.0435, 0.0002, -0.0021, 0.0044, -0.6192, 0.9991],
        'sample': 27, 'timestep':1
    },

    #Sample 27, Timestep 2
   {
        'predicted': [-0.0080, 0.0018, 0.0380, -0.0295, -0.0852, 0.0209, 0.0000],
        'ground_truth': [0.0027, 0.0478, -0.0026, 0.0010, 0.0082, -0.6088, 0.9959],
        'sample': 27, 'timestep':2
    },

    #Sample 28, Timestep 1
   {
        'predicted': [-0.0047, -0.0004, -0.0021, -0.0033, 0.0166, -0.0178, 0.0000],
        'ground_truth': [-0.0096, 0.0235, -0.0024, -0.0001, -0.0021, -0.9147, 0.0621],
        'sample': 28, 'timestep':1
    },

    #Sample 28, Timestep 2
   {
        'predicted': [-0.0047, -0.0037, 0.0312, -0.0180, -0.0497, -0.0081, 0.9961],
        'ground_truth': [-0.0093, 0.0172, 0.0048, -0.0059, -0.0032, -0.8765, 0.0594],
        'sample': 28, 'timestep':2
    },

    #Sample 29, Timestep 1
   {
        'predicted': [-0.0076, 0.0407, -0.0117, 0.0292, 0.0200, 0.0870, 0.0000],
        'ground_truth': [0.0033, -0.0178, 0.0018, 0.0039, 0.0018, -0.3363, 0.0662], 
        'sample': 29, 'timestep':1
    },

    #Sample 29, Timestep 2
   {
        'predicted': [0.0000, 0.0018, -0.0057, 0.0120, 0.0200, 0.0032, 0.0000],
        'ground_truth': [-0.0033, -0.0194, 0.0055, -0.0015, 0.0005, -0.3148, 0.9978],
        'sample': 29, 'timestep':2
    },
        #Sample 30, Timestep 1
   {
        'predicted': [0.0246, 0.0086, -0.0080, -0.0116, -0.0095, -0.0033, 0.0000],
        'ground_truth': [0.0008, 0.0465, 0.0095, 0.0060, -0.0083, -0.6037, 1.0000],
        'sample': 30, 'timestep':1
    },

    #Sample 30, Timestep 2
   {
        'predicted': [0.0058, -0.0092, 0.0021, -0.0250, -0.0122, -0.0017, 0.9961],
        'ground_truth': [-0.0097, 0.0469, 0.0043, -0.0005, -0.0055, -0.5842, 1.0000],
        'sample': 30, 'timestep':2
    }
]

def get_hardcoded_data():
    """Return hardcoded predictions and ground truth"""
    openvla_preds = []
    ground_truths = []
    
    for data in HARDCODED_PREDICTIONS:
        # Skip empty arrays
        if len(data['predicted']) == 0 or len(data['ground_truth']) == 0:
            continue
            
        openvla_preds.append(np.array(data['predicted']))
        ground_truths.append(np.array(data['ground_truth']))
    
    print(f"✅ Loaded {len(openvla_preds)} valid predictions from {len(HARDCODED_PREDICTIONS)} total entries")
    return openvla_preds, ground_truths

def add_sample_from_terminal(predicted, ground_truth, sample, timestep):
    """Add a new sample from terminal output"""
    new_sample = {
        'predicted': predicted,
        'ground_truth': ground_truth,
        'sample': sample,
        'timestep': timestep
    }
    HARDCODED_PREDICTIONS.append(new_sample)
    print(f"✅ Added Sample {sample}, Timestep {timestep}")

def _format_vec(vec):
    arr = np.array(vec, dtype=float).flatten().tolist()
    return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"

def print_baseline_range(start_sample: int, end_sample: int, instruction: str = "pick up the object and place it in the bowl"):
    by_sample = {}
    for entry in HARDCODED_PREDICTIONS:
        s = int(entry.get('sample'))
        if start_sample <= s <= end_sample:
            by_sample.setdefault(s, []).append(entry)

    if not by_sample:
        print(f"❌ No entries found for samples {start_sample}-{end_sample}")
        return

    for s in sorted(by_sample.keys()):
        entries = sorted(by_sample[s], key=lambda e: int(e.get('timestep', 0)))

        print(f"\n📸 Sample {s}:")
        print(f"Instruction: {instruction}")

        for e in entries:
            timestep = int(e.get('timestep', 0))
            pred = e.get('predicted', [])
            gt = e.get('ground_truth', [])

            print(f"Debug: gt_action type={type(gt)}, shape={np.array(gt).shape if len(gt) else 'N/A'}")
            print(f"Debug: after conversion shape={np.array(gt).reshape(-1).shape if len(gt) else 'N/A'}")
            print(f"Debug: final shape={np.array(gt).reshape(-1).shape if len(gt) else 'N/A'}")
            print(f"🖼️  Image {timestep}: im_{timestep}.jpg")
            print(f"✅ Ground Truth: {_format_vec(gt)}")
            print(f"🎯 Predicted:   {_format_vec(pred)}")

# Example of how to add samples as you see them:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print hardcoded OpenVLA baseline results")
    parser.add_argument("--start", type=int, default=None, help="Start sample index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End sample index (inclusive)")
    args = parser.parse_args()

    if args.start is not None and args.end is not None:
        print_baseline_range(args.start, args.end)
    else:
        print("📝 Hardcoded Data Template")
        print("Copy-paste values from your terminal output:")
        print("")
        print("Example usage:")
        print("add_sample_from_terminal(")
        print("    predicted=[-0.0083, 0.0177, 0.0380, 0.0420, -0.0611, 0.0999, 0.0000],")
        print("    ground_truth=[-0.0291, 0.0698, 0.0007, -0.0083, 0.0017, -0.6787, 0.0609],")
        print("    sample=1, timestep=1")
        print(")")
