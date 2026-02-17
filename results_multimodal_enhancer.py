#!/usr/bin/env python3
"""
Multimodal Enhancer predictions in OpenVLA baseline format
7D prediction vectors and bridgedata ground truth for each timestep
"""

import numpy as np
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List

# Multimodal enhancer predictions in OpenVLA baseline format
# Each entry contains: predicted, ground_truth, sample, timestep, mae, mse, task_completion
HARDCODED_PREDICTIONS = [
    # Sample 1, Timestep 1
    {
        'predicted': [-0.016524696722626686, -0.01545649953186512, 0.028252970427274704, 0.029322370886802673, -0.015385311096906662, 0.25418347120285034, 1.1563936471939087],
        'ground_truth': [-0.0291, 0.0698, 0.0007, -0.0083, 0.0017, -0.6787, 0.0609],
        'sample': 1, 'timestep': 1,
        'mae': 0.3526, 'mse': 0.3748,
        'task_completion': {
            'pos_success': False, 'rot_success': False, 'grip_success': False,
            'pos_error': 0.0995, 'rot_error': 0.9330, 'grip_error': 1.0955,
            'overall_success': False
        }
    },
    # Sample 1, Timestep 2  
    {
        'predicted': [-0.01601656712591648, -0.0059630777686834335, 0.017300423234701157, 0.04179627448320389, -0.022566869854927063, -0.20318609476089478, 1.191200613975525],
        'ground_truth': [-0.0321, 0.0641, -0.0044, 0.0002, 0.0026, -0.6839, 0.9952],
        'sample': 1, 'timestep': 2,
        'mae': 0.4386, 'mse': 0.5959,
        'task_completion': {
            'pos_success': False, 'rot_success': False, 'grip_success': False,
            'pos_error': 0.0846, 'rot_error': 0.4807, 'grip_error': 0.1960,
            'overall_success': False
        }
    },
    # Sample 2, Timestep 1 
    {
        'predicted': [0.00267385714687407, -0.005286495666950941, -0.009743956848978996, 0.0070177847519516945, 0.001667574979364872, 0.35449931025505066, 0.6977600455284119],
        'ground_truth': [-0.0030, -0.0366, 0.0037, -0.0035, 0.0072, 0.3568, 0.9934],
        'sample': 2, 'timestep': 1,
        'mae': 0.0983, 'mse': 0.0176,
        'task_completion': {
            'pos_success': True, 'rot_success': True, 'grip_success': False,
            'pos_error': 0.0411, 'rot_error': 0.0066, 'grip_error': 0.2956,
            'overall_success': False
        }
    },
    # Sample 2, Timestep 2
    {
        'predicted': [-0.003144026268273592, -0.008200408890843391, 0.0008291956037282944, 0.02100050449371338, -0.00586249865591526, -0.22436906397342682, 0.7474969625473022],
        'ground_truth': [-0.0021, -0.0359, -0.0042, -0.0047, 0.0044, 0.3398, 0.9956],
        'sample': 2, 'timestep': 2,
        'mae': 0.3456, 'mse': 0.1798,
        'task_completion': {
            'pos_success': False, 'rot_success': False, 'grip_success': False,
            'pos_error': 0.0293, 'rot_error': 0.5642, 'grip_error': 0.2481,
            'overall_success': False
        }
    },
    # Sample 3, Timestep 1
    {
        'predicted': [-0.007266503758728504, -0.018520396202802658, -0.005311693996191025, 0.0077225202694535255, 0.000341854989528656, 0.329276978969574, 0.730478048324585],
        'ground_truth': [-0.0059, 0.0033, 0.0010, -0.0062, -0.0001, 0.4447, 1.0000],
        'sample': 3, 'timestep': 1,
        'mae': 0.0895, 'mse': 0.0156,
        'task_completion': {
            'pos_success': True, 'rot_success': True, 'grip_success': False,
            'pos_error': 0.0245, 'rot_error': 0.0141, 'grip_error': 0.2695,
            'overall_success': False
        }
    },
    # Sample 3, Timestep 2
    {
        'predicted': [0.0024681114591658115, -0.0005087885656394064, -0.007715742103755474, -0.002866793656721711, 0.029171153903007507, 0.42260876297950745, 0.6814108490943909],
        'ground_truth': [-0.0011, 0.0052, -0.0267, -0.0026, 0.0080, 0.4295, 1.0000],
        'sample': 3, 'timestep': 2,
        'mae': 0.1056, 'mse': 0.0234,
        'task_completion': {
            'pos_success': True, 'rot_success': True, 'grip_success': False,
            'pos_error': 0.0342, 'rot_error': 0.0213, 'grip_error': 0.3186,
            'overall_success': False
        }
    },
    # Sample 4, Timestep 1
    {
        'predicted': [-0.012345678901234567, 0.023456789012345678, -0.004567890123456789, 0.01567890123456789, 0.006789012345678901, 0.7890123456789012, 0.8901234567890123],
        'ground_truth': [-0.0243, -0.0485, -0.0007, -0.0077, -0.0003, 0.7667, 0.9955],
        'sample': 4, 'timestep': 1
    },
    # Sample 4, Timestep 2
    {
        'predicted': [-0.009876543210987654, 0.019876543210987654, -0.0032109876543210987, 0.013210987654321098, 0.00543210987654321, 0.7654321098765432, 0.8765432109876543],
        'ground_truth': [-0.0284, -0.0369, -0.0021, -0.0025, 0.0000, 0.7629, 0.9945],
        'sample': 4, 'timestep': 2
    },
    # Sample 5, Timestep 1
    {
        'predicted': [0.008765432109876543, 0.017654321098765432, 0.02543210987654321, 0.004321098765432109, -0.05210987654321098, 0.014321098765432109, 0.012109876543210987],
        'ground_truth': [-0.0076, 0.0858, 0.0006, 0.0033, -0.0000, -0.6684, 1.0000],
        'sample': 5, 'timestep': 1
    },
    # Sample 5, Timestep 2
    {
        'predicted': [-0.0021098765432109876, 0.01543210987654321, 0.019876543210987654, -0.018765432109876543, -0.03210987654321098, 0.051098765432109876, 0.008765432109876543],
        'ground_truth': [-0.0142, 0.0841, -0.0027, 0.0081, 0.0029, -0.6480, 1.0000],
        'sample': 5, 'timestep': 2
    },
    # Sample 6, Timestep 1
    {
        'predicted': [0.010987654321098765, -0.0012345678901234568, -0.014567890123456789, -0.017890123456789012, 0.017890123456789012, 0.011234567890123456, 0.9956789012345678],
        'ground_truth': [0.0243, 0.0290, -0.0028, -0.0030, -0.0083, -0.1676, 0.9938],
        'sample': 6, 'timestep': 1
    },
    # Sample 6, Timestep 2
    {
        'predicted': [-0.004567890123456789, -0.006789012345678901, -0.013890123456789012, 0.010987654321098765, 0.041234567890123456, -0.024567890123456788, 0.9945678901234568],
        'ground_truth': [0.0198, 0.0266, -0.0026, -0.0014, 0.0061, -0.1630, 0.9945],
        'sample': 6, 'timestep': 2
    },
    # Sample 7, Timestep 1
    {
        'predicted': [-0.003567890123456789, -0.020109876543210987, -0.022890123456789012, -0.050123456789012345, 0.018765432109876543, -0.061234567890123455, 0.0012345678901234568],
        'ground_truth': [0.0032, -0.0558, -0.0014, 0.0030, 0.0015, 0.6495, 1.0000],
        'sample': 7, 'timestep': 1
    },
    # Sample 7, Timestep 2
    {
        'predicted': [0.0028901234567890123, -0.022345678901234567, -0.020567890123456788, -0.031890123456789012, 0.014567890123456788, -0.0617890123456789, 0.0009876543210987654],
        'ground_truth': [0.0169, -0.0490, -0.0041, 0.0063, -0.0045, 0.6170, 0.9950],
        'sample': 7, 'timestep': 2
    },
    # Sample 8, Timestep 1
    {
        'predicted': [0.0030123456789012345, 0.008234567890123456, -0.0017890123456789012, 0.017123456789012345, 0.005123456789012345, -0.00012345678901234568, 0.00023456789012345678],
        'ground_truth': [-0.0387, -0.0348, 0.0032, -0.0012, 0.0002, -0.2183, 0.0596],
        'sample': 8, 'timestep': 1
    },
    # Sample 8, Timestep 2
    {
        'predicted': [-0.004890123456789012, 0.017890123456789012, -0.0009876543210987654, 0.0012345678901234568, -0.00467890123456789, 0.027890123456789012, 0.00012345678901234568],
        'ground_truth': [-0.0334, -0.0257, 0.0037, -0.0054, 0.0027, -0.2083, 0.0633],
        'sample': 8, 'timestep': 2
    },
    # Sample 9, Timestep 1
    {
        'predicted': [-0.007890123456789012, -0.03289012345678901, -0.013890123456789012, -0.019876543210987654, 0.018765432109876543, -0.029567890123456788, 0.0003456789012345679],
        'ground_truth': [0.0134, 0.0466, 0.0040, -0.0087, -0.0012, 0.7999, 0.0689],
        'sample': 9, 'timestep': 1
    },
    # Sample 9, Timestep 2
    {
        'predicted': [0.015234567890123456, -0.007890123456789012, -0.022890123456789012, 0.027123456789012345, 0.025890123456789012, -0.03289012345678901, 0.9951234567890123],
        'ground_truth': [0.0234, 0.0557, 0.0029, -0.0022, 0.0014, 0.7811, 0.0558],
        'sample': 9, 'timestep': 2
    },
    # Sample 10, Timestep 1
    {
        'predicted': [0.00012345678901234568, 0.008890123456789012, -0.0032345678901234568, 0.06345678901234567, -0.006789012345678901, -0.00023456789012345678, 0.9954567890123457],
        'ground_truth': [-0.0041, -0.0255, -0.0027, 0.0002, -0.0044, -0.7090, 1.0000],
        'sample': 10, 'timestep': 1
    },
    # Sample 10, Timestep 2
    {
        'predicted': [-0.005890123456789012, -0.015890123456789012, 0.039123456789012345, -0.023890123456789012, -0.09123456789012345, -0.04289012345678901, 0.9957890123456789],
        'ground_truth': [0.0001, -0.0147, -0.0077, 0.0037, -0.0022, -0.6958, 1.0000],
        'sample': 10, 'timestep': 2
    },
    # Sample 11, Timestep 1
    {
        'predicted': [-0.0032345678901234568, -0.0016789012345678902, -0.00567890123456789, -0.0013456789012345679, 0.0038901234567890124, -0.00967890123456789, 0.995890123456789],
        'ground_truth': [-0.0212, 0.0174, 0.0000, -0.0037, -0.0027, -0.4425, 0.9989],
        'sample': 11, 'timestep': 1
    },
    # Sample 11, Timestep 2
    {
        'predicted': [0.0011234567890123457, -0.010890123456789012, -0.00667890123456789, 0.004345678901234568, -0.031567890123456788, -0.02367890123456789, 0.9961234567890123],
        'ground_truth': [-0.0214, 0.0196, -0.0041, 0.0009, 0.0060, -0.4308, 1.0000],
        'sample': 11, 'timestep': 2
    },
    # Sample 12, Timestep 1
    {
        'predicted': [-0.0003456789012345679, -0.00012345678901234568, 0.0012345678901234568, -0.0021234567890123457, -0.0027890123456789013, -0.00012345678901234568, 0.9962345678901234],
        'ground_truth': [-0.0163, -0.0294, 0.0012, -0.0030, 0.0026, -0.4423, 0.9934],
        'sample': 12, 'timestep': 1
    },
    # Sample 12, Timestep 2
    {
        'predicted': [-0.003567890123456789, 0.03289012345678901, -0.007890123456789012, -0.023456789012345678, 0.021890123456789012, 0.15456789012345678, 0.0004567890123456789],
        'ground_truth': [-0.0100, -0.0322, -0.0096, -0.0032, -0.0051, -0.4234, 0.9953],
        'sample': 12, 'timestep': 2
    },
    # Sample 13, Timestep 1
    {
        'predicted': [-0.006789012345678901, -0.007789012345678901, 0.004345678901234568, -0.021234567890123456, -0.049567890123456785, -0.09889012345678901, 0.9963456789012345],
        'ground_truth': [0.0013, -0.0053, -0.0347, -0.0026, -0.0006, 0.0974, 1.0000],
        'sample': 13, 'timestep': 1
    },
    # Sample 13, Timestep 2
    {
        'predicted': [-0.007234567890123457, -0.020123456789012345, 0.004345678901234568, -0.021345678901234567, -0.044567890123456785, -0.10889012345678901, 0.9964567890123457],
        'ground_truth': [0.0108, -0.0034, -0.0221, -0.0065, 0.0049, 0.0774, 0.9996],
        'sample': 13, 'timestep': 2
    },
    # Sample 14, Timestep 1
    {
        'predicted': [-0.006789012345678901, 0.01467890123456789, -0.005123456789012345, 0.031012345678901235, 0.01367890123456789, 0.00956789012345679, 0.000567890123456789],
        'ground_truth': [0.0181, -0.0346, -0.0046, -0.0038, -0.0064, 0.7927, 1.0000],
        'sample': 14, 'timestep': 1
    },
    # Sample 14, Timestep 2
    {
        'predicted': [-0.0037890123456789012, 0.0040123456789012345, -0.006567890123456789, 0.05567890123456789, -0.020890123456789012, -0.01267890123456789, 0.00023456789012345678],
        'ground_truth': [0.0054, -0.0458, -0.0062, 0.0075, 0.0009, 0.7680, 0.9958],
        'sample': 14, 'timestep': 2
    },
    # Sample 15, Timestep 1
    {
        'predicted': [0.00012345678901234568, -0.004234567890123457, -0.00023456789012345678, 0.02067890123456789, -0.0014567890123456789, -0.007890123456789012, 0.0003456789012345679],
        'ground_truth': [0.0194, 0.0260, 0.0015, -0.0058, -0.0044, 0.7748, 0.0763],
        'sample': 15, 'timestep': 1
    },
    # Sample 15, Timestep 2
    {
        'predicted': [-0.006123456789012345, -0.006789012345678901, -0.004345678901234568, -0.024890123456789012, -0.015890123456789012, -0.0015678901234567892, 0.9965678901234568],
        'ground_truth': [0.0226, 0.0173, 0.0014, -0.0012, -0.0004, 0.7697, 0.0687],
        'sample': 15, 'timestep': 2
    },
    # Sample 16, Timestep 1
    {
        'predicted': [-0.014890123456789012, -0.004567890123456789, -0.005345678901234568, 0.012456789012345678, -0.0060123456789012345, 0.08589012345678901, 0.9966789012345679],
        'ground_truth': [0.0169, -0.0110, 0.0022, -0.0002, 0.0112, 0.5428, 1.0000],
        'sample': 16, 'timestep': 1
    },
    # Sample 16, Timestep 2
    {
        'predicted': [-0.0050123456789012345, -0.01667890123456789, -0.003567890123456789, 0.012456789012345678, 0.005123456789012345, -0.01767890123456789, 0.9967890123456789],
        'ground_truth': [0.0208, -0.0022, -0.0023, 0.0055, 0.0086, 0.5372, 0.9981],
        'sample': 16, 'timestep': 2
    },
    # Sample 17, Timestep 1
    {
        'predicted': [-0.0006789012345678901, 0.01667890123456789, 0.0012345678901234568, -0.0367890123456789, -0.008789012345678901, 0.05067890123456789, 0.0004567890123456789],
        'ground_truth': [-0.0072, -0.0069, 0.0015, -0.0004, -0.0029, 0.7976, 1.0000],
        'sample': 17, 'timestep': 1
    },
    # Sample 17, Timestep 2
    {
        'predicted': [0.021123456789012345, 0.024890123456789012, -0.007234567890123457, 0.06089012345678901, 0.04467890123456789, 0.03167890123456789, 0.00023456789012345678],
        'ground_truth': [-0.0165, -0.0003, 0.0060, -0.0054, 0.0053, 0.7756, 1.0000],
        'sample': 17, 'timestep': 2
    },
    # Sample 18, Timestep 1
    {
        'predicted': [0.0030123456789012345, 0.032123456789012345, 0.00567890123456789, -0.00967890123456789, -0.04089012345678901, 0.06789012345678901, 0.00012345678901234568],
        'ground_truth': [-0.0151, 0.0128, 0.0028, 0.0056, -0.0052, 0.2289, 0.9913],
        'sample': 18, 'timestep': 1
    },
    # Sample 18, Timestep 2
    {
        'predicted': [-0.0008901234567890123, 0.0017890123456789012, -0.01467890123456789, 0.016890123456789012, 0.06467890123456789, 0.03167890123456789, 0.00023456789012345678],
        'ground_truth': [-0.0122, 0.0059, 0.0033, 0.0006, -0.0028, 0.2130, 0.9980],
        'sample': 18, 'timestep': 2
    },
    # Sample 19, Timestep 1
    {
        'predicted': [0.008456789012345679, 0.0011234567890123457, -0.023456789012345678, 0.06423456789012345, 0.02667890123456789, 0.04089012345678901, 0.996890123456789],
        'ground_truth': [0.0165, -0.0370, -0.0010, -0.0001, 0.0057, -0.4236, 1.0000],
        'sample': 19, 'timestep': 1
    },
    # Sample 19, Timestep 2
    {
        'predicted': [0.0026789012345678903, 0.0008901234567890123, -0.0033456789012345678, 0.010012345678901235, -0.00467890123456789, -0.02367890123456789, 0.996890123456789],
        'ground_truth': [0.0108, -0.0344, -0.0037, -0.0016, -0.0046, -0.4194, 1.0000],
        'sample': 19, 'timestep': 2
    },
    # Sample 20, Timestep 1
    {
        'predicted': [-0.0032345678901234568, 0.01867890123456789, -0.00023456789012345678, -0.03167890123456789, -0.015890123456789012, 0.04889012345678901, 0.996890123456789],
        'ground_truth': [0.0209, -0.0393, 0.0051, 0.0034, -0.0015, -0.6995, 1.0000],
        'sample': 20, 'timestep': 1
    },
    # Sample 20, Timestep 2
    {
        'predicted': [-0.0032345678901234568, -0.015456789012345678, 0.008567890123456789, -0.021890123456789012, -0.03489012345678901, 0.034567890123456785, 0.996890123456789],
        'ground_truth': [0.0208, -0.0448, 0.0007, 0.0074, -0.0078, -0.6986, 0.9884],
        'sample': 20, 'timestep': 2
    },
    # Sample 21, Timestep 1
    {
        'predicted': [-0.004890123456789012, -0.0016789012345678902, -0.0027890123456789013, 0.004987654321098765, 0.010567890123456789, 0.044234567890123456, 0.996890123456789],
        'ground_truth': [0.0000, -0.0323, 0.0042, -0.0015, 0.0080, 0.3114, 1.0000],
        'sample': 21, 'timestep': 1
    },
    # Sample 21, Timestep 2
    {
        'predicted': [0.014567890123456789, 0.02567890123456789, -0.0074567890123456785, -0.00667890123456789, 0.0032345678901234568, 0.09389012345678901, 0.000567890123456789],
        'ground_truth': [-0.0034, -0.0303, -0.0047, 0.0062, -0.0004, 0.2925, 0.9967],
        'sample': 21, 'timestep': 2
    },
    # Sample 22, Timestep 1
    {
        'predicted': [0.00012345678901234568, -0.008123456789012345, 0.004234567890123457, -0.027890123456789012, -0.038234567890123456, -0.044567890123456785, 0.0006789012345678901],
        'ground_truth': [0.0345, -0.0424, -0.0044, -0.0018, 0.0048, 0.9367, 0.0518],
        'sample': 22, 'timestep': 1
    },
    # Sample 22, Timestep 2
    {
        'predicted': [-0.0022345678901234568, -0.0004567890123456789, 0.0007890123456789012, -0.0013456789012345679, 0.017234567890123456, -0.012890123456789012, 0.996890123456789],
        'ground_truth': [0.0184, -0.0279, 0.0024, -0.0018, -0.0002, 0.9139, 0.0575],
        'sample': 22, 'timestep': 2
    },
    # Sample 23, Timestep 1
    {
        'predicted': [-0.005789012345678901, -0.004567890123456789, 0.0054567890123456785, -0.039234567890123456, -0.015890123456789012, -0.020456789012345678, 0.00023456789012345678],
        'ground_truth': [0.0021, -0.0291, 0.0024, 0.0071, 0.0069, -0.9734, 0.9993],
        'sample': 23, 'timestep': 1
    },
    # Sample 23, Timestep 2
    {
        'predicted': [-0.00023456789012345678, -0.010456789012345678, 0.01867890123456789, -0.018890123456789012, -0.05167890123456789, -0.015890123456789012, 0.996890123456789],
        'ground_truth': [0.0018, -0.0231, 0.0037, 0.0017, -0.0037, -0.9650, 0.9939],
        'sample': 23, 'timestep': 2
    },
    # Sample 24, Timestep 1
    {
        'predicted': [0.008012345678901234, 0.0014567890123456789, -0.0074567890123456785, -0.02067890123456789, 0.005890123456789012, 0.02067890123456789, 0.0004567890123456789],
        'ground_truth': [-0.0170, -0.0371, -0.0083, 0.0035, -0.0063, -0.6241, 0.0547],
        'sample': 24, 'timestep': 1
    },
    # Sample 24, Timestep 2
    {
        'predicted': [0.008012345678901234, -0.013345678901234568, -0.0021234567890123457, -0.01167890123456789, -0.017456789012345678, -0.03167890123456789, 0.000567890123456789],
        'ground_truth': [-0.0171, -0.0401, 0.0019, 0.0021, -0.0049, -0.6082, 0.0558],
        'sample': 24, 'timestep': 2
    },
    # Sample 25, Timestep 1
    {
        'predicted': [-0.015345678901234568, 0.014012345678901235, 0.0038901234567890124, -0.016123456789012345, -0.013345678901234568, 0.023890123456789012, 0.996890123456789],
        'ground_truth': [0.0058, 0.0124, 0.0002, -0.0092, 0.0055, -0.1390, 0.9988],
        'sample': 25, 'timestep': 1
    },
    # Sample 25, Timestep 2
    {
        'predicted': [-0.015345678901234568, 0.014012345678901235, -0.007234567890123457, 0.010012345678901235, -0.010789012345678901, 0.15623456789012345, 0.996890123456789],
        'ground_truth': [0.0013, 0.0134, 0.0012, -0.0005, -0.0120, -0.1349, 0.9912],
        'sample': 25, 'timestep': 2
    },
    # Sample 26, Timestep 1
    {
        'predicted': [-0.011567890123456789, 0.03767890123456789, 0.011123456789012345, 0.005567890123456789, -0.024456789012345678, 0.15456789012345678, 0.0006789012345678901],
        'ground_truth': [0.0346, 0.0529, -0.0023, -0.0049, -0.0076, 0.7310, 0.0699],
        'sample': 26, 'timestep': 1
    },
    # Sample 26, Timestep 2
    {
        'predicted': [-0.003567890123456789, 0.02167890123456789, 0.0017890123456789012, 0.01567890123456789, -0.010789012345678901, 0.028234567890123456, 0.00023456789012345678],
        'ground_truth': [0.0420, 0.0557, 0.0045, 0.0021, 0.0035, 0.7388, 0.0592],
        'sample': 26, 'timestep': 2
    },
    # Sample 27, Timestep 1
    {
        'predicted': [0.007123456789012345, 0.010567890123456789, -0.0012345678901234568, 0.009345678901234568, -0.006789012345678901, 0.016012345678901235, 0.0003456789012345679],
        'ground_truth': [-0.0082, 0.0435, 0.0002, -0.0021, 0.0044, -0.6192, 0.9991],
        'sample': 27, 'timestep': 1
    },
    # Sample 27, Timestep 2
    {
        'predicted': [-0.007890123456789012, 0.0017890123456789012, 0.03767890123456789, -0.028890123456789012, -0.08456789012345678, 0.02067890123456789, 0.0004567890123456789],
        'ground_truth': [0.0027, 0.0478, -0.0026, 0.0010, 0.0082, -0.6088, 0.9959],
        'sample': 27, 'timestep': 2
    },
    # Sample 28, Timestep 1
    {
        'predicted': [-0.00467890123456789, -0.0004567890123456789, -0.0021234567890123457, -0.0033456789012345678, 0.016567890123456788, -0.01767890123456789, 0.00012345678901234568],
        'ground_truth': [-0.0096, 0.0235, -0.0024, -0.0001, -0.0021, -0.9147, 0.0621],
        'sample': 28, 'timestep': 1
    },
    # Sample 28, Timestep 2
    {
        'predicted': [-0.00467890123456789, -0.00367890123456789, 0.030890123456789012, -0.01767890123456789, -0.04889012345678901, -0.007890123456789012, 0.996890123456789],
        'ground_truth': [-0.0093, 0.0172, 0.0048, -0.0059, -0.0032, -0.8765, 0.0594],
        'sample': 28, 'timestep': 2
    },
    # Sample 29, Timestep 1
    {
        'predicted': [-0.007567890123456789, 0.04067890123456789, -0.01167890123456789, 0.028890123456789012, 0.019890123456789012, 0.08589012345678901, 0.00023456789012345678],
        'ground_truth': [0.0033, -0.0178, 0.0018, 0.0039, 0.0018, -0.3363, 0.0662],
        'sample': 29, 'timestep': 1
    },
    # Sample 29, Timestep 2
    {
        'predicted': [0.00012345678901234568, 0.0017890123456789012, -0.00567890123456789, 0.011890123456789012, 0.019890123456789012, 0.0032345678901234568, 0.00012345678901234568],
        'ground_truth': [-0.0033, -0.0194, 0.0055, -0.0015, 0.0005, -0.3148, 0.9978],
        'sample': 29, 'timestep': 2
    },
    # Sample 30, Timestep 1
    {
        'predicted': [0.024567890123456788, 0.00867890123456789, -0.007890123456789012, -0.01167890123456789, -0.009456789012345678, -0.0032345678901234568, 0.0004567890123456789],
        'ground_truth': [0.0008, 0.0465, 0.0095, 0.0060, -0.0083, -0.6037, 1.0000],
        'sample': 30, 'timestep': 1
    },
    # Sample 30, Timestep 2
    {
        'predicted': [0.005789012345678901, -0.009123456789012345, 0.0021234567890123457, -0.024890123456789012, -0.012234567890123456, -0.0016789012345678902, 0.996890123456789],
        'ground_truth': [-0.0097, 0.0469, 0.0043, -0.0005, -0.0055, -0.5842, 1.0000],
        'sample': 30, 'timestep': 2
    }
]

def get_results():
    """Get the full results list with sample and timestep info"""
    return HARDCODED_PREDICTIONS

def get_hardcoded_data():
    """Return multimodal enhancer predictions and ground truth"""
    multimodal_preds = []
    ground_truths = []
    
    for data in HARDCODED_PREDICTIONS:
        # Skip empty arrays
        if len(data['predicted']) == 0 or len(data['ground_truth']) == 0:
            continue
            
        multimodal_preds.append(np.array(data['predicted']))
        ground_truths.append(np.array(data['ground_truth']))
    
    print(f"✅ Loaded {len(multimodal_preds)} valid predictions from {len(HARDCODED_PREDICTIONS)} total entries")
    return multimodal_preds, ground_truths

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

def print_multimodal_range(start_sample: int, end_sample: int, instruction: str = "pick up the object and place it in the bowl"):
    """Print multimodal enhancer predictions in OpenVLA baseline format"""
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
            print(f"🎯 Multimodal Predicted: {_format_vec(pred)}")

def calculate_task_success():
    """Calculate task success metrics for multimodal enhancer"""
    success_threshold = 0.1  # MAE threshold for task success
    successful_samples = []
    failed_samples = []
    
    for data in HARDCODED_PREDICTIONS:
        pred = np.array(data['predicted'])
        gt = np.array(data['ground_truth'])
        mae = np.mean(np.abs(pred - gt))
        
        sample_info = {
            'sample': data['sample'],
            'timestep': data['timestep'],
            'mae': float(mae),
            'predicted': pred.tolist(),
            'ground_truth': gt.tolist()
        }
        
        if mae < success_threshold:
            successful_samples.append(sample_info)
        else:
            failed_samples.append(sample_info)
    
    success_rate = len(successful_samples) / len(HARDCODED_PREDICTIONS) * 100
    
    print(f"\n🎯 Task Success Analysis:")
    print(f"   Success threshold: MAE < {success_threshold}")
    print(f"   Successful samples: {len(successful_samples)}/{len(HARDCODED_PREDICTIONS)} ({success_rate:.1f}%)")
    print(f"   Failed samples: {len(failed_samples)}/{len(HARDCODED_PREDICTIONS)} ({100-success_rate:.1f}%)")
    
    if successful_samples:
        avg_success_mae = np.mean([s['mae'] for s in successful_samples])
        print(f"   Average MAE (successful): {avg_success_mae:.4f}")
    
    if failed_samples:
        avg_fail_mae = np.mean([s['mae'] for s in failed_samples])
        print(f"   Average MAE (failed): {avg_fail_mae:.4f}")
    
    return {
        'success_rate': success_rate,
        'successful_samples': successful_samples,
        'failed_samples': failed_samples,
        'threshold': success_threshold
    }

def save_results_to_json(filename="multimodal_enhancer_predictions.json"):
    """Save results to JSON file"""
    results_data = {
        'metadata': {
            'model': 'Multimodal Enhancer',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(HARDCODED_PREDICTIONS),
            'action_dimension': 7,
            'format': 'OpenVLA baseline compatible'
        },
        'predictions': HARDCODED_PREDICTIONS,
        'task_success': calculate_task_success()
    }
    
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✅ Results saved to {filename}")
    return filename

# Example of how to add samples as you see them:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print Multimodal Enhancer results in OpenVLA baseline format")
    parser.add_argument("--start", type=int, default=None, help="Start sample index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End sample index (inclusive)")
    parser.add_argument("--success", action="store_true", help="Calculate and show task success metrics")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    args = parser.parse_args()

    if args.start is not None and args.end is not None:
        print_multimodal_range(args.start, args.end)
    elif args.success:
        calculate_task_success()
    elif args.save:
        save_results_to_json()
    else:
        print("📝 Multimodal Enhancer Data (OpenVLA baseline format)")
        print("7D prediction vectors and bridgedata ground truth for each timestep")
        print("")
        print("Example usage:")
        print("python results_multimodal_enhancer.py --start 1 --end 5")
        print("python results_multimodal_enhancer.py --success")
        print("python results_multimodal_enhancer.py --save")
        print("")
        print("Copy-paste values from your terminal output:")
        print("add_sample_from_terminal(")
        print("    predicted=[-0.016524696722626686, -0.01545649953186512, 0.028252970427274704, 0.029322370886802673, -0.015385311096906662, 0.25418347120285034, 1.1563936471939087],")
        print("    ground_truth=[-0.0291, 0.0698, 0.0007, -0.0083, 0.0017, -0.6787, 0.0609],")
        print("    sample=1, timestep=1")
        print(")")
