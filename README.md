# Goodness-of-Fit and Parameter Uncertainty

This repository contains code for estimating multidimensional Item Response Theory (IRT) models using **amortized variational inference (AVI)**, as well as likelihood-based **Fisher scoring**.

The code is used to demonstrate that performing a **single step of Fisher scoring** is sufficient to compute:
- goodness-of-fit statistics, and  
- parameter standard errors.

---

## Repository Contents

- **example.py**  
  A minimal working example in which data are simulated, model parameters are estimated using AVI, and a single Fisher scoring step is performed.  
  This script can easily be adapted to apply the same approach to other datasets.

- **avi.py**  
  Contains all code required to estimate multidimensional IRT (MIRT) models using amortized variational inference.

- **onstep.py**  
  Contains the implementation of a single Fisher scoring step, including the computation of parameter standard errors.

- **helpers.py**  
  Provides auxiliary helper functions used throughout the codebase.

- **Qmatrices/**  
  A directory containing `.csv` files with factor structures used in the paper.  
  This includes both:
  - factor structures used in the simulation studies, and  
  - literature-based factor structures for the NPI.

- **requirements.txt**  
  A list of all libraries required to run the code in this repository.

---

## Example Usage

To run a quick example on simulated data, execute the following commands:

```bash
pip install -r requirements.txt
python3 example.py
