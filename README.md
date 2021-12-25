# Option pricing

This repository contains a few notebooks on numerical methods for financial mathematics, especially for option pricing. The focus is on implementation and numerics. Topics covered include stochastic differential equations, the Black-Scholes model, and the multilevel Monte Carlo method.

## Content

- [Brownian motion and Euler-Maruyama](notebooks/Brownian_motion_and_Euler-Maruyama.ipynb) - Introducing the Wiener process, the geometric Brownian motion, as well as the Euler-Maruyama method for solving stochastic differential equations.
- [Strong and weak convergence](notebooks/Strong_and_weak_convergence.ipynb) - Confirming the strong and weak convergence orders for the Euler-Maruyama and the Milstein method.
- [Monte Carlo European option](notebooks/Monte_Carlo_European_Option.ipynb) - Using Monte Carlo methods to price European options and investigating two sources of error.
- [Multilevel Monte Carlo method](notebooks/Multilevel_Monte_Carlo.ipynb) - Introduction and analysis of the Multilevel Monte Carlo method by replicating the results in [Multilevel Monte Carlo path simulation](https://ora.ox.ac.uk/objects/uuid:d9d28973-94aa-4179-962a-28bcfa8d8f00/download_file?safe_filename=2007OMI06.pdf&file_format=application%2Fpdf&type_of_work=Working+paper) by Michael B. Giles.