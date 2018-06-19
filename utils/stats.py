def pearson_correlation(x1, x2, eps=1e-8):
    r""" See: https://github.com/vishwakftw/pytorch/blob/dcceaa5f5d282e53ebf3be45493dfa9f130a7376/torch/nn/functional.py
    Returns Pearson coefficient between 1D-tensors x1 and x2
    ..math ::
        \text{correlation} = \dfrac{\bar{x_1} \cdot \bar{x_2}}
                                {\max(\Vert \bar{x_1} \Vert _2 \cdot \Vert \bar{x_2} \Vert _2, \epsilon)}
        \text{where } \bar{z} \text{ denotes the mean-reduced version of } z
    Args:
        x1 (Variable): First input (1D).
        x2 (Variable): Second input (of size matching x1).
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input: :math: `(\ast_1, \ast_2)`
        - Output: :math: `(1)`
    Example::
        >>> input1 = autograd.Variable(torch.randn(128))
        >>> input2 = autograd.Variable(torch.randn(128))
        >>> output = F.pearson_correlation(input1, input2)
        >>> print(output)
    """
    assert x1.dim() == 1, "Input must be 1D matrix / vector."
    assert x1.size() == x2.size(), "Input sizes must be equal."
    x1_bar = x1 - x1.mean()
    x2_bar = x2 - x2.mean()
    dot_prod = x1_bar.dot(x2_bar)
    norm_prod = x1_bar.norm(2) * x2_bar.norm(2)
    return dot_prod / norm_prod.clamp(min=eps)
