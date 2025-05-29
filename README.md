# StructureDiscovery
This repository contains the code accompanying the NeurIPS 2025 paper:

**"A Derandomization Framework for Structure Discovery: Applications in Neural Networks and Beyond"**  
Authors: Nikos Tsikouras, Christos Tzamos, Ioannis Mitliagkas, Yorgos Pantis

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
│
└── README.md                     # Project overview and instructions
```
