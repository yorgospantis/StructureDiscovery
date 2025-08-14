<div align="center">
<h1>A Derandomization Framework for Structure Discovery: Applications in Neural Networks and Beyond</h1>

**Nikos Tsikouras<sup>1,2</sup>, Christos Tzamos<sup>1,2</sup>, Ioannis Mitliagkas<sup>2,3,4,5,6</sup>, Yorgos Pantis<sup>1,2</sup>**

<sup>1</sup>Department of Informatics and Telecommunications, National and Kapodistrian University of Athens, Greece<br>
<sup>2</sup>Archimedes, Athena Research Center, Greece<br>
<sup>3</sup>Université de Montréal<br>
<sup>4</sup>Mila - Quebec AI Institute<br>
<sup>5</sup>Canada CIFAR AI chair<br>
<sup>6</sup>Google DeepΜind<br>

*Authors listed in contrubution order*

[![Paper](https://img.shields.io/badge/ICLR-2026-blue?logo=book&logoColor=white)](https://openreview.net/forum?id=yourpaperid)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


</div>

## Abstract
Understanding neural networks (NNs) theoretically remains a significant challenge. The work of [MHPG+22](https://iclr.cc/virtual/2023/poster/11421) shows that under some conditions, a two-layer NN can discover low-rank structure, which can be used to understand its generalization behavior. In this paper, we zoom in on this problem and extend their findings to a more general setting. We allow (a) NNs of arbitrary size and depth, (b) with all parameters trainable, (c) under any loss function, and (d) tiny regularization. At the core of our approach is a key *derandomization* Lemma, which states that optimizing the function **E**<sub>**x**</sub> [**g**<sub>θ</sub>(**W** **x** + **b**)] converges to a point where **W = 0**, under mild conditions. The fundamental nature of this Lemma directly explains structure discovery and has immediate applications in other domains including an end-to-end approximation for MAXCUT, and computing Johnson-Lindenstrauss embeddings.


## Citation

If you use this work, please cite:

```bibtex
@inproceedings{derandomization2026, 
  title={A Derandomization Framework for Structure Discovery: Applications in Neural Networks and Beyond}, 
  author={Tsikouras, Nikos and Tzamos, Christos and Mitliagkas, Ioannis and Pantis, Yorgos},
  booktitle={The Four-teenth International Conference on Learning Representations}, 
  year={2026}
}
```

## Project Structure
```
# StructureDiscovery Project Directory

StructureDiscovery/
│
├── NeuralNetorks/
│   ├── activations.py            # Activation functions 
│   ├── model.py                  # Neural network model architecture
│   ├── train.py                  # Training pipeline and routines
│   ├── plot.py                   # Training curve and evaluation visualizations
│   └── main.py                   # Main file that needs all the above
│
├── JohnsonLindenstrauss/
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

## Acknowledgments
This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.

