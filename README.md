# CSCI_6212_12_Fall2025_Project3

### Project 3 of Design and Analysis of Algorithms on Bi-connectivity

## Experimental Analysis of Biconnectivity algorithm:

# Overview
The following is the problem statement that is to be analyzed:

Given a graph G, check if the graph is biconnected or not. If it is not, identify all the articulation points.  The algorithm should run in linear time. Use the given input sets as tests.

## Inputs (n) Tested for:

The following `n` values have been used in the experiment:

```python
n_values = [100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000]

```
## Prerequisites
In order to run the analysis script, Python 3.x environment is needed with the following libraries:

* **`numpy`**: this is used for mathematical operations, especially for functions such as `np.log2` and `np.polyfit`.
* **`matplotlib`**: this is used to plot the experimental and theoretical results.
* **`time`**: this is used to calculate the runtime of the function and to obtain the experimental time complexity.

The installation can be done using `pip`:
```bash
pip install numpy matplotlib
```
## Procedure to execute the code

1. In order to get the program running, a suitable terminal is required through which the project folder can be navigated and accessed.

```bash
cd DesignAndAnalysis_Algorithms
```

2. Run the Python script:

```bash
python project3.py
```
3. The script performs the following tasks:

   * It generates random graph inputs for each value in n_values with controlled edge density to simulate different connectivity scenarios.
   * It computes the experimental runtime of the articulation point detection function for each graph size and stores the results.
   * It scales the theoretical values (based on n+m) to align with the experimental runtime by calculating appropriate scaling constants using linear regression.
   * It prints a table comparing experimental runtimes against theoretical operation counts and the scaled theoretical runtime values.
   * It also displays a graph comparing experimental runtime vs scaled theoretical runtime for a better understanding of the growth rates exhibited by the two runtimes.

---
