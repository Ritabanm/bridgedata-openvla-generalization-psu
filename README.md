# OpenVLA Enhancement Framework

Enhancing Vision-Language-Action models through multimodal neural networks, game theory ensembles, and reinforcement learning.

## 🎯 Overview

Multiple approaches to improve OpenVLA predictions on BridgeData robotics tasks:

- **Multimodal Enhancer**: Attention-based neural network with data augmentation
- **Game Theory Ensemble**: Cooperative game theory using Shapley values  
- **Advanced RL**: PPO-based action correction and policy refinement

## 📁 Core Files

| File | Purpose | Key Features |
|------|---------|--------------|
| **`openvla-baseline.py`** | OpenVLA baseline implementation | SOTA VLA model predictions |
| **`multimodal_enhancer.py`** | Multimodal neural enhancement | Attention mechanisms, data augmentation |
| **`game_theory_enhancer.py`** | Game theory ensemble | Shapley values, cooperative learning |
| **`advanced_rl_experiments.py`** | RL-based enhancement | PPO action correction, safety layers |
| **`experiments.py`** | Comprehensive testing | Cross-validation, statistical analysis |

## 🚀 Quick Setup

```bash
# Environment
conda create -n openvla-psu python=3.10
conda activate openvla-psu

# Dependencies
pip install torch numpy scikit-learn matplotlib pillow tqdm scipy

# Required data
# Place: baseline_100_samples_results.json in repo root
```

## 💻 Usage

```bash
# Run individual methods
python multimodal_enhancer.py
python game_theory_enhancer.py  
python advanced_rl_experiments.py

# Comprehensive evaluation
python experiments.py
```

## 📊 Results

- **Multimodal Enhancer**: 20-40% improvement on complex tasks
- **Game Theory**: 10-25% improvement through ensemble diversity
- **Advanced RL**: 5-15% improvement through policy refinement

## 🔧 Key Parameters

```python
# Multimodal
HIDDEN_DIM = 256
AUGMENT_FACTOR = 5

# Game Theory  
NUM_PLAYERS = 5

# Advanced RL
PPO_EPOCHS = 10
```

## 🚨 Troubleshooting

- **MPS Error**: Auto-detects CPU fallback
- **Missing Data**: Check `baseline_100_samples_results.json`
- **Memory Issues**: Use CPU mode or reduce batch size

## 🎯 Quick Start

```bash
conda create -n openvla-psu python=3.10 && conda activate openvla-psu
pip install torch numpy scikit-learn matplotlib pillow tqdm scipy
python multimodal_enhancer.py
```

**Runtime**: 5-15 minutes on CPU, 2-5 minutes on GPU.

---

## 📄 Citation

```bibtex
@misc{openvla_enhancement,
  title={OpenVLA Enhancement Framework: Multimodal and Game Theory Approaches},
  author={Ritaban Mitra},
  year={2026},
  url={https://github.com/ritabanm/bridgedata-openvla-generalization-psu}
}
```
