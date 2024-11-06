# Queens Solver

A Mixed Integer Linear Programme (MILP) solver for the Queens game on LinkedIn.

## Extracting the Game

Given a screenshot such as the following 

![Sample Screenshot](SampleScreenshot.png)

We first need to extract the game in order to solve it. 

## MILP Solver

Representing each cell on the board as a binary value, with 0 representing an empty cell, and 1 representing the presence of a Queen, we can unravel the $n \times n$ board into a vector ${\bf x} \in \{0, 1\}^{\otimes n^2}$.

The rule dictating that each row and column can only contain a single Queen can be expressed as a linear constraint over ${\bf x}$