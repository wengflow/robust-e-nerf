import torch


def gradient(outputs, inputs, retain_graph=None, create_graph=False):
    """
    TODO:
        1. Better description
        2. Gradient scaling during automatic mixed precision training

    Args:
        outputs (torch.Tensor / Sequence of torch.Tensor):
            (Sequence of) Output data
        inputs (torch.Tensor / Sequence of torch.Tensor):
            (Sequence of) Input data that requires gradients (does not need to
            be a leaf tensor)
    Returns:
        gradient (torch.Tensor / Tuple of torch.Tensor):
            Gradient in the same form as the inputs
    """
    if isinstance(outputs, torch.Tensor):
        grad_outputs = torch.ones_like(outputs, requires_grad=False)
    else:   # sequence of `torch.Tensor`
        grad_outputs = tuple(
            torch.ones_like(output, requires_grad=False)
            for output in outputs
        )

    gradient = torch.autograd.grad(
        outputs, inputs, grad_outputs, retain_graph, create_graph
    )
    if isinstance(inputs, torch.Tensor):
        gradient = gradient[0]  # remove the redundant tuple packing

    return gradient


def jacobian(output, inputs, retain_graph=None, create_graph=False):
    """
    TODO:
        1. Better description
        2. Gradient scaling during automatic mixed precision training

    Args:
        output (torch.Tensor):
            Output data of shape (..., K), where K is the output dimension
        inputs (torch.Tensor / Sequence of torch.Tensor):
            (Sequence of) Input data that requires gradients (does not need to
            be a leaf tensor)
    Returns:
        jacobian (torch.Tensor / Tuple of torch.Tensor):
            Jacobian with shape (*inputs.shape, K) / len(inputs)-tuple of
            jacobian, each with shape (*inputs[index].shape, K)
    """
    # compute the Jacobian by computing the gradients for each element in the
    # output dimension
    output_dim = output.shape[-1]

    # wrap tensor inputs in a tuple
    is_inputs_tensor = isinstance(inputs, torch.Tensor)
    if is_inputs_tensor:
        inputs = (inputs, )

    # `retain_graph=True` is required in order to compute gradients iteratively
    gradients = (
        gradient(outputs=output[..., index],
                 inputs=inputs,
                 retain_graph=True,
                 create_graph=create_graph)
        for index in range(output_dim)
    )
    jacobian = tuple(
        torch.stack(input_gradients, dim=-1)
        for input_gradients in zip(*gradients)
    )

    if is_inputs_tensor:
        jacobian = jacobian[0]  # remove the redundant tuple packing

    # detach the computational graph retained on the output, which is used for
    # the computation of the Jacobian, if it is not required
    if retain_graph is False or (retain_graph is None and not create_graph):
        output.detach_()

    return jacobian
