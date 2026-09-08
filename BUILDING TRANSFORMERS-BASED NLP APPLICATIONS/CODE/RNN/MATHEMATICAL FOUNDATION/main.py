import numpy as np

# ---------------------------------------------------------------
# 1. Activation Functions
# ---------------------------------------------------------------
def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------------------------------------------------------------
# 2. RNN definitions
# ---------------------------------------------------------------

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

    # HT calculations H_t=ϕ(X_t·W_xh + H_t−1·W_hh + b_h) where ϕ is hiperbolic tangent
    X_tW_xh = np.dot(X_t, W_xh)
    H_t1Whh = np.dot(H_prev, W_hh)
    H_t = tanh(X_tW_xh + H_t1Whh + b_h)

    # Ot calculations O_t=ϕ(H_t·W_ho + b_o) where ϕ is logistic sigmoid
    Ht_W_ho = np.dot(H_t, W_ho)
    O_t = sigmoid(Ht_W_ho + bo)

    return H_t, O_t

# ---------------------------------------------------------------
# 3. Environment setup
# ---------------------------------------------------------------

np.random.seed(42)

batch_size = 1
seq_len = 5
input_dim = 8   # e.g., 2 features per step
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

# ---------------------------------------------------------------
# 4. Run simulation
# ---------------------------------------------------------------

print("--- Running RNN Forward pass with simulation steps. ---")

for t in range(seq_len):
    X_t = X_sequence[t] # Get input for current timestep
    
    # Run one RNN step
    H_prev, O_t = rnn_step(X_t, H_prev, W_xh, W_hh, b_h, W_ho, b_o)
    
    print(f"\n[Timestep {t+1}]")
    print(f"  Input X_t shape: {X_t.shape}")
    print(f"  Hidden State H_{t+1}: {np.round(H_prev, 4)}")
    print(f"  Output O_{t+1} (Probability): {np.round(O_t, 4)}")