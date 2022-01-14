# CI-Project 2021-2022

This project presents the implementation of a Genetic Algorithm to solve the 0-1 knapsack problem.

## Project structure

The project is structured as follows:
```
.
├── best_solutions
├── data
├── figures
├── report
├── solutions
├── times
├── requirements.txt
└── solver.py
```

The `best_solutions` folder contains the optimal solutions found with the provided `magic_d` function. This files are used by the solver to compare the GA solution with the optimal one. The `data` folder contains all the problem files provided to us. The `figures` folder contains the plots with the fitness evolution for each problem file. The `solution` folder contains the files with the best found solution for each problem file. The `times` file contains the execution times for different executions of the problem files.

We also include a `requirements.txt` file since we use additional libraries such as `matplotlib` to plot the results and `tqdm` to show the execution progress. To install the required packages please create a new python environment and after activating it run the following command:

```bash
pip install -r requirements.txt
```

We would like to point out that we slightly modified some of the default code in the `solver.py` to be able to read the files in the `best_solutions` folder. Nevertheless, its functionality has been unchanged and so it can be used as its original version with the following command:
```bash
python solver.py ./data/ninja_X_Y
```

Also, its execution will append the time used to the corresponding file in the `times` folder and if desired it can plot the figure as well. The latter is deactivated by default and should be uncommented from the code if desired.
