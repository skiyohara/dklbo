# dklbo

## Overview
**dklbo** is a software to perform **Bayesian optimization (BO)** using **Deep Kernel Learning (DKL)** for materials search.  
This software is not directly applicable to individual cases because the neural network part of DKL depends on the specific task.  

Therefore, this package serves as an **example to demonstrate how to use DKL**.

## Instructions
This package assumes the use of **graph-based neural networks** and modifies 's `ExactGP` class in `gpytorch` to create a custom class **`ExactGP_graph`**, which allows **`Data`** class in pytorch_geometric to be used as input.  

Additionally, to facilitate handling **high-entropy alloys with multiple sites**, a custom data structure called **`SiteGraph`**, which inherits from `Data` class in pytorch_geometric, is used.  
This enables handling multiple sites individually within a **single Data object**.

## Sample Code
### 1. **`/example/search_bandgap.py`**
- Unzip /example/datasets/calculation/cifs.zip
- Searches for the material with the **largest band gap** among **922 oxides**.  
- Uses **CGCNN** for the neural network.  
  - 📄 [Reference: CGCNN](https://doi.org/10.1103/PhysRevLett.120.145301)

### 2. **`/example/search_expbandgap.py`**
- Searches for the material with the **largest band gap** among **610 organic-inorganic hybrid perovskites (ABX₃)**.  
- Uses a **graph-based neural network**, where convolution is applied to each **A-site, B-site, and X-site** separately.  
- The **`SiteGraph`** structure is utilized to manage site-specific information in ABX₃.

## Installation
```bash
# Clone the repository
git clone https://github.com/skiyohara/dklbo.git

# Required packages
pytorch
pytorch_geometric
pytorch_scatter
gpytorch
pymatgen
