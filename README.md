# ML-sys-opt
this repo is for ML system optimization programming assignment


# Parallel K-Means Clustering Assignment

This project implements the K-Means clustering algorithm in two modes: **Sequential** (single-core) and **Parallel** (multi-core using `multiprocessing`). It benchmarks the execution time and accuracy of both approaches to demonstrate the speedup achieved through parallelization.

## Prerequisites
* Python 3.8 or higher

##  Setup & Installation

Follow these steps to set up your environment and run the code.

### 1. specific Clone or Download the Repository
Navigate to the project directory in your terminal:
```bash
cd path/to/your/project-folder
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

#### Windows
```bash
python -m venv venv
```
#### Mac OS
```bash
python3 -m venv venv
```

3. Activate the Virtual Environment

#### Windows:
```bash

.\venv\Scripts\activate
```
#### MacOS / Linux:

```bash
source .venv/bin/activate
```
(You will see (venv) appear at the start of your terminal line indicating it is active.)

4. Install Dependencies
Install the required libraries (NumPy, Scikit-Learn, SciPy) using the requirements.txt file:

```bash
pip install -r requirements.txt
```
How to Run
Once the installation is complete, you can run the main script. Ensure your script file is named main.py.

```bash
python main.py
```