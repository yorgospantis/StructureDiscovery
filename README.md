# StructureDiscovery
This repository contains the code accompanying the NeurIPS 2025 paper:

**"A Derandomization Framework for Structure Discovery: Applications in Neural Networks and Beyond"**  
Authors: Nikos Tsikouras, Christos Tzamos, Ioannis Mitliagkas, Yorgos Pantis

## Abstract
Understanding neural networks (NNs) theoretically remains a significant challenge. The work of [MHPG+22](https://iclr.cc/virtual/2023/poster/11421) shows that under some conditions, a two-layer NN can discover low-rank structure, which can be used to understand its generalization behavior. In this paper, we zoom in on this problem and extend their findings to a more general setting. We allow (a) NNs of arbitrary size and depth, (b) with all parameters trainable, (c) under any loss function, and (d) tiny regularization. At the core of our approach is a key *derandomization* Lemma, which states that optimizing the function \\( \\mathbb{E}_{\\mathbf{x}} [g_{\\theta}(\\mathbf{W}\\mathbf{x} + \\mathbf{b})] \\) converges to a point where \\( \\mathbf{W} = \\mathbf{0} \\), under mild conditions. The fundamental nature of this Lemma directly explains structure discovery and has immediate applications in other domains including an end-to-end approximation for MAXCUT, and computing Johnson-Lindenstrauss embeddings.

![formula](https://latex.codecogs.com/png.image?\\dpi{120} \\mathbb{E}_{\\mathbf{x}}%20[g_{\\theta}(\\mathbf{W}\\mathbf{x}%20+%20\\mathbf{b})]%20\\Rightarrow%20\\mathbf{W}%20=%20\\mathbf{0})


## Citation

If you use this work, please cite:

```bibtex
@inproceedings{tsikourasderandomization2025, 
  title={A Derandomization Framework for Structure Discovery: Applications in Neural Networks and Beyond}, 
  author={Tsikouras, Nikos and Tzamos, Christos and Mitliagkas, Ioannis and Pantis, Yorgos},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}, 
  year={2025}
}
```

## Project Structure
```
# StructureDiscovery Project Directory

StructureDiscovery/
│
├── NNs/
│   ├── activations.py            # Activation functions 
│   ├── model.py                  # Neural network model architecture
│   ├── train.py                  # Training pipeline and routines
│   ├── plot.py                   # Training curve and evaluation visualizations
│   └── main.py                   # Main file that needs all the above
│
├── JL/
│   ├── model.py                  # Model architecture
│   ├── train.py                  # Optimization loop for training the distortion model
│   ├── plot.py                   # Plots of distortion and variance evolution
│   └── main.py                   # Main file that needs all the above
│
├── MAXCUT/
│   ├── utilities.py              # Helper functions: exact MAXCUT, e.g. graph generation
│   ├── optimizer.py              # Optimization routine for MAXCUT using gradients
│   ├── plot.py                   # Plot sigma and cut values over iterations
│   └── main.py                   # Main file that needs all the above
│
├── LICENSE                       # MIT open-source license
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview and instructions
```
