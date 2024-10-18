# IPM-LSTM: A Learning-Based Interior Point Method for Solving Nonlinear Programs
This repo is an implementation of the NeurIPS 2024 paper: IPM-LSTM: A Learning-Based Interior Point Method for Solving Nonlinear Programs.
## Introduction
We focus on solving the following NLP:
$$
\begin{aligned}
        & \underset{x \in \mathbb{R}^{n}} {\text{min}} &&  f(x)      \\
        & \; \text{s.t.} &&  h(x) = 0    \\
        & &&  x \geq 0 
\end{aligned}
$$
