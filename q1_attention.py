import numpy as np

def softmax(x):
    # subtract max for numerical stability
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Q: Query matrix   (shape: seq_len × d_k)
    K: Key matrix     (shape: seq_len × d_k)
    V: Value matrix   (shape: seq_len × d_v)

    Returns:
        attention_weights: softmax-normalized similarity scores
        context_vector: weighted sum of values
    """

    # Step 1: compute raw attention scores (QK^T)
    scores = np.dot(Q, K.T)

    # Step 2: scale by sqrt(d_k)
    d_k = Q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: apply softmax to get normalized weights
    attention_weights = softmax(scaled_scores)

    # Step 4: compute context vector = weights × V
    context_vector = np.dot(attention_weights, V)

    return attention_weights, context_vector


# Example usage (test)
if __name__ == "__main__":
    Q = np.random.rand(3, 4)
    K = np.random.rand(3, 4)
    V = np.random.rand(3, 6)

    att_wt, ctx = scaled_dot_product_attention(Q, K, V)
    print("Attention Weights:\n", att_wt)
    print("Context Vector:\n", ctx)
