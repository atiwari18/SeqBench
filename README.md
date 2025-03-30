# SeqBench
SeqBench is a project that benchmarks three different neural networks for sequence modeling: a Vanilla RNN, and two variants of the RNN one that uses Padding (to maximum sequence length) and another that uses truncation (down to minimum sequence length). The task is given an input sequence to predict a single output.

## Benchmarking Pipeline
1. First all the models are trained on a training data found in the `.npy` files in the directory.
2. During training the models are evaluated based on Mean Squared Error (MSE) across epochs these losses are collected and then presented in a graph at the end.
3. Models are then evaluated on the training set to see how well they generalize to new data.

## Results
| Model |  Advantages  | Disadvantages |
|:-----|:--------|:------|
| Vanilla RNN   | Low Loss, Learns Effectively, Good generalization | With very long sequences it is bound to struggle due to loss of information courtesy of the vanishing gradient problem.  |
| Truncated Model   | Prevents overfitting by reducing sequence length.  |   Discards valuable information necessary therefore causing the model to have struggles learning.  |
| Padded Model   | Sequences retain length thereby preserving complete information. |   Introduce additional noise and bias which hinders the ability to learn effectively.  |

![Training Graph](https://github.com/atiwari18/SeqBench/blob/main/Question%20%232%20Losses.png)
