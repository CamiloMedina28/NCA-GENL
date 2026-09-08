# RNN practical implementations

## Mathematical process

To understand how the RNN works over time, a simple mathematical model was implemented with random values. It also allows to change predefined values to analyse how it changes.

### Main functions

The two activation functions that were defined are sigmoid and hiperbolic tangent. Using the advantages of numpy to model them. 

The two equations that are expolained on [Sequencing models](../../NOTES/Sequencing_Modeling.md) are implemented using `np.dot` for the matrix dot product. 

```python
def rnn_step(X_t, H_prev, W_xh, W_hh, b_h, W_ho, bo):
    """RNN mathematical step - single forward

    Args:
        X_t: Input at current timestep t (shape: [batch size, input dimensions])
        H_prev: Memory vector carried from previous time step (shape: [batch size, hidden dimensions])
        W_xh: Input to hidden weight matrix (shape: [input_dim, hidden_dim])
        W_hh: Hidden-to-hidden weight matrix (shape: [hidden_dim, hidden_dim])
        b_h: Hidden bias (shape: [1, hidden_dim])
        W_ho: Hidden-to-output weight matrix (shape: [hidden_dim, output_dim])
        bo: Output bias (shape: [1, output_dim])

    Returns:
        - H_t: Current hidden state
        - O_t: Output at current timestep
    """

    ...

    return H_t, O_t
```

### Environment set up

Environment set up is crucial to allow reproducibility of the experiment several times and to obtain the same results. Nevertheless, the environment can be set with customized values by the user. 

```python
np.random.seed(42)

batch_size = 1
seq_len = 3
input_dim = 2   # e.g., 2 features per step
hidden_dim = 4  # size of memory vector
output_dim = 1  # binary prediction at each step

X_sequence = np.random.randn(seq_len, batch_size, input_dim)

# Initialize Weights and Biases randomly
W_xh = np.random.randn(input_dim, hidden_dim) * 0.1
W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
b_h  = np.zeros((1, hidden_dim))

W_ho = np.random.randn(hidden_dim, output_dim) * 0.1
b_o  = np.zeros((1, output_dim))

# Initial hidden state (H_0) initialized to zeros
H_prev = np.zeros((batch_size, hidden_dim))
```

- np.random_seed(42): Makes sure that the random generated values are always the same.
- batch_size = 1: Only proccesses one sequence at a time
- seq_lenght = n: Sequence contains n steps during the simulation, can contain as many as necessary and as many as machine's processing capacities allow. 
- input_dim = n: The inputs received by the RNN, usually it receives one. However, as explained before can contain as many as necesary due to the flexibility it offers.
- hidden_dim = n: The hidden state vector/inner memory. The greates this number, the capacity of learning complex patterns. 
- output_dim = 1: In each step the exit will only have one dimension, for instance, a probability. 

**Entry data generation**

as a hypothetical case, let us take:
batch_size = 1
seq_len = 3
input_dim = 2   # e.g., 2 features per step
```python
X_sequence = np.random.randn(seq_len, batch_size, input_dim)
```
a possible `X_sequence` will look as: 
```json
[
  [[ 0.49, -0.13 ]],   # Step t=1: Día 1 -> [Temp, Humidity]
  [[ 0.64,  1.52 ]],   # Step t=2: Día 2 -> [Temp, Humidity]
  [[-0.23, -0.23 ]]    # Step t=3: Día 3 -> [Temp, Humidity]
]
```
As seen,
- There are three steps T.
- Batch size of one indicates that only one array enters at a time
- input_dim indicates that two measures are entering.

