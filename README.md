# ArtSAGENet
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This is the **official implementation** of the paper:

> **Graph Neural Networks for Knowledge Enhanced Visual Representation of Paintings**  
> Athanasios Efthymiou, Stevan Rudinac, Monika Kackovic, Marcel Worring, and Nachoem Wijnberg  
> *In Proceedings of the 29th ACM International Conference on Multimedia (MM '21)*  
> [📄 Paper (ACM DL)](https://doi.org/10.1145/3474085.3475586)

---

## Table of Contents
- [Model Overview](#model-overview)
- [Tasks](#tasks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

---

## Model Overview

<p align="center">
  <img src="assets/graph_pipeline.png" alt="Graph-Structured Visual Representation Pipeline" width="750"/>
</p>

**Figure 1**: Illustration of the fine art analysis pipeline. Each painting is treated as a node in a graph, where edges represent semantic relationships such as shared artist, school, genre, or period. Visual features are extracted via a CNN and enhanced through context-aware message passing with a GNN.

<p align="center">
  <img src="assets/artsagenet_architecture.png" alt="ArtSAGENet Architecture" width="750"/>
</p>

**Figure 2**: The ArtSAGENet architecture. Visual features are extracted using ResNet-152 (frozen except for the last bottleneck), and semantic context is modeled using GraphSAGE. The resulting multimodal embeddings are trained jointly in a multi-task setting for several downstream tasks.

ArtSAGENet is a novel multimodal architecture that integrates:
- **Graph Neural Networks (GNNs)** for semantic relationship modeling
- **Convolutional Neural Networks (CNNs)** for visual feature extraction
- **Multi-task learning** for joint optimization across related tasks

---

## Tasks

ArtSAGENet supports multiple fine art analysis tasks:

- **Style Classification**: Predict the artistic style/movement
- **Artist Attribution**: Identify the artist who created the artwork
- **Creation Period Estimation**: Estimate when the artwork was created
- **Tag Prediction**: Multi-label classification for artwork tags

---

## Installation

### Requirements
- Python 3.9+
- PyTorch 1.7+
- PyTorch Geometric
- torchvision
- scikit-learn
- numpy
- pandas
- Pillow

### Setup
```bash
# Clone the repository
git clone https://github.com/thefth/ArtSAGENet.git
cd ArtSAGENet

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Training ArtSAGENet (Multi-task)
```bash
python artsagenet/train.py \
  --data_dir graph_data/ \
  --labels_task1 labels_styles \
  --labels_task2 labels_artists \
  --labels_task3 labels_timelines \
  --task_type classification \
  --epochs 50 \
  --batch_size 16
```

### Training ArtSAGENet (Single-task)
```bash
python artsagenet/train.py \
  --data_dir graph_data/ \
  --labels_task1 labels_styles \
  --no-multi_task \
  --single_task 1 \
  --task_type classification \
  --epochs 50
```

### Training CNN Baselines
```bash
python cnn_baselines/train.py \
  --model resnet-152 \
  --data_path dataset.pkl \
  --image_dir WikiArt/Dataset/ \
  --epochs 100 \
  --batch_size 16
```

### Training GNN Baselines
```bash
# GraphSAGE
python gnn_baselines/graphsage.py \
  --data_dir graph_data/ \
  --task labels_styles \
  --task_type classification \
  --epochs 1000

# ClusterGCN
python gnn_baselines/clustergcn.py \
  --data_dir graph_data/ \
  --task labels_styles \
  --task_type classification \
  --epochs 1000
```

For more training options and examples, see the documentation in each module.

---

## Dataset

The WikiArt datasets are available in the `Dataset/` directory. Three subsets are provided:

| Dataset | Artworks | Artists | Styles | Tasks |
|---------|----------|---------|--------|-------|
| **WikiArt<sup>Full</sup>** | 75,921 | 750 | 20 | Style, Artist, Timeframe, Tags |
| **WikiArt<sup>Modern</sup>** | 45,865 | 462 | 13 | Style, Artist, Date |
| **WikiArt<sup>Artists</sup>** | 17,785 | 23 | 12 | Style, Artist, Timeframe |

### Getting the Images

The annotations are provided in this repository, but you need to download the actual images:

1. Download images from the [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)
2. Place the image folder in your desired location
3. Update the `--image_dir` or `--images` argument when running training scripts

For detailed dataset documentation, see [Dataset/README.md](Dataset/README.md).

---

## Repository Structure
```
ArtSAGENet/
├── artsagenet/          # ArtSAGENet implementation
│   ├── train.py         # Main training script
│   ├── artsagenet.py    # Model architecture
│   ├── loader.py        # Custom data loader
│   ├── helpers.py       # Training/testing functions
│   └── utils.py         # Utility functions
├── cnn_baselines/       # CNN baseline models
│   ├── train.py         # Training script
│   ├── helpers.py       # Helper functions
│   ├── dataloader.py    # Data loading utilities
│   └── utils.py         # Utility functions
├── gnn_baselines/       # GNN baseline models
│   ├── graphsage.py     # GraphSAGE implementation
│   ├── clustergcn.py    # ClusterGCN implementation
│   ├── graphsaint.py    # GraphSAINT implementation
│   ├── sign.py          # SIGN implementation
│   └── utils.py         # Shared utilities
├── Dataset/             # WikiArt dataset annotations
│   ├── README.md        # Dataset documentation
│   ├── wikiart_full.csv
│   ├── wikiart_modern.csv
│   └── wikiart_artists.csv
├── requirements.txt     # Python dependencies
├── LICENSE             # License file
└── README.md           # This file
```

---

## Citation

If you use the WikiArt datasets or code from this repository in your research, please cite:
```bibtex
@inproceedings{efthymiou2021graph,
  title={Graph Neural Networks for Knowledge Enhanced Visual Representation of Paintings},
  author={Efthymiou, Athanasios and Rudinac, Stevan and Kackovic, Monika and Worring, Marcel and Wijnberg, Nachoem},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={3710--3719},
  year={2021}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) for the artwork images

---

**Questions or Issues?** Please contact the corresponding author or open an issue on our [GitHub repository](https://github.com/thefth/ArtSAGENet/issues).