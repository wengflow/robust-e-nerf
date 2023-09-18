import torch
import roma


def randperm_manual_seed(n, seed):
    """`torch.randperm()` using the given seed, indp. of the default RNG"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(n, generator=generator)


def normalize_range(input, min, max):
    return (input - min) / (max - min)


def bool_mean(bool_tensor):
    return bool_tensor.to(torch.get_default_dtype()).mean()


def lerp_uniform(input, upsampling_factor):
    """
    Args:
        input (torch.Tensor): Input data of shape (N, ...) & floating type
        upsampling_factor (int): Upsampling factor
    Returns:
        upsampled_input (torch.Tensor): Input data uniformly upsampled
                                        along the first dimension with shape
                                        ((N-1) * upsampling_factor + 1, ...)
    """
    N = input.shape[0]
    remaining_shape = input.shape[1:]                                           # ie. (...)
    last_input = input[-1, ...].unsqueeze(dim=0)                                # (1, ...)

    weight_shape = [ upsampling_factor ] + [ 1 ] * input.dim()
    weight = torch.arange(                                                      # (upsampling_factor)
        0.0, 1.0, 1 / upsampling_factor, dtype=input.dtype, device=input.device
    )
    weight = weight.view(*weight_shape)                                         # (upsampling_factor, <input.dim() no. of 1>)

    input = input.unsqueeze(dim=0)                                              # (1, N, ...)
    input = torch.lerp(                                                         # (upsampling_factor, N-1, ...)
        input=input[:, :-1, ...], end=input[:, 1:, ...], weight=weight
    )
    input = input.transpose(0, 1)                                               # (N-1, upsampling_factor, ...)
    # TODO: lerp along the 2nd dimension to enable view() instead of reshape()
    input = input.reshape((N-1) * upsampling_factor, *remaining_shape)          # ((N-1) * upsampling_factor, ...)
    input = torch.cat((input, last_input), dim=0)                               # ((N-1) * upsampling_factor + 1, ...)

    upsampled_input = input
    return upsampled_input


def slerp_uniform(input, upsampling_factor):
    """
    Args:
        input (torch.Tensor): Input quaternions of shape (N, ..., 4) &
                              floating point type
        upsampling_factor (int): Upsampling factor
    Returns:
        upsampled_input (torch.Tensor): Input quaternions uniformly upsampled
                                        along the first dimension with shape
                                        ((N-1) * upsampling_factor + 1, ..., 4)
    """
    assert input.shape[-1] == 4
    N = input.shape[0]
    remaining_shape = input.shape[1:]                                           # ie. (..., 4) 
    last_input = input[-1, ...].unsqueeze(dim=0)                                # (1, ..., 4)

    steps = torch.arange(                                                       # (upsampling_factor)
        0.0, 1.0, 1 / upsampling_factor, dtype=input.dtype, device=input.device
    )
    input = roma.unitquat_slerp(                                                # (upsampling_factor, N-1, ... 4)
        q0=input[:-1, ...], q1=input[1:, ...], steps=steps, shortest_path=True
    )
    input = input.transpose(0, 1)                                               # (N-1, upsampling_factor, ..., 4)
    input = input.reshape((N-1) * upsampling_factor, *remaining_shape)          # ((N-1) * upsampling_factor, ..., 4)
    input = torch.cat((input, last_input), dim=0)                               # ((N-1) * upsampling_factor + 1, ..., 4)

    upsampled_input = input
    return upsampled_input


def unitquat_to_full_rotvec(quat):
    """
    Variant of `roma.unitquat_to_rotvec()` returning rotation vectors with
    angles within [0, 2pi], instead of restricting it to [0, pi] (acute angles)
    
    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
            No normalization is applied before computation.
    Returns:
        batch of rotation vectors (...x3 tensor).
    """
    quat, batch_shape = roma.internal.flatten_batch_dims(quat, end_dim=-2)
    # We perform a copy to support auto-differentiation.
    quat = quat.clone()
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L1006-L1073
    angle = 2 * torch.atan2(torch.norm(quat[:, :3], dim=1), quat[:, 3])
    small_angle = (torch.abs(angle) <= 1e-3)
    large_angle = ~small_angle

    num_rotations = len(quat)
    scale = torch.empty(num_rotations, dtype=quat.dtype, device=quat.device)
    scale[small_angle] = (2 + angle[small_angle] ** 2 / 12 +
                          7 * angle[small_angle] ** 4 / 2880)
    scale[large_angle] = (angle[large_angle] /
                          torch.sin(angle[large_angle] / 2))

    rotvec = scale[:, None] * quat[:, :3]
    return roma.internal.unflatten_batch_dims(rotvec, batch_shape)


def unitquat_slerp(q0, q1, steps, shortest_path=False):
    """
    Variant of `roma.unitquat_slerp()` supporting distinct steps for each
    quaternion pairs in the batch.
    
    Args: 
        q0, q1 (Ax4 tensor): batch of unit quaternions (A may contain multiple
                             dimensions).
        steps (tensor of shape B, A): interpolation steps, 0.0 corresponding
                                      to q0 and 1.0 to q1.
        shortest_path (boolean): if True, interpolation will be performed along
                                 the shortest path on SO(3).
    Returns: 
        batch of interpolated quaternions (BxAx4 tensor).

    NOTE:
        1. When considering quaternions as rotation representations,
           one should keep in mind that spherical interpolation is not
           necessarily performed along the shortest arc, depending on the sign
           of `torch.sum(q0*q1,dim=-1)`.

           This function can also work when:
               1. only B is empty
               2. only A is empty
               3. both A & B are empty
        2. As of RoMa v1.2.7, `roma.unitquat_to_rotvec()` always returns
           rotation vectors with angles within [0, pi] radians. Hence, the
           `shortest_path` argument of `roma.unitquat_slerp()` is effectively
           always true regardless of its specified value. Here, we use
           `unitquat_to_full_rotvec()` as a hotfix.
        3. As of RoMa v1.2.7, the quaternion flipping for shortest path
           interpolation is incorrectly reversed & returns zero quaternion for
           `q1` when the rotation angle between `q0` & `q1` is pi radians. The
           hotfix is included here.

        References:
            1. https://github.com/naver/roma/blob/22806dfb43329b9bf1dd2cead7e96720330e3151/roma/mappings.py#L248
            2. https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp
            3. https://github.com/naver/roma/issues/3

    Implementation adapted from `roma.unitquat_slerp()`
    https://github.com/naver/roma/blob/22806dfb43329b9bf1dd2cead7e96720330e3151/roma/utils.py#L255
    """
    if shortest_path:
        # Flip some quaternions to ensure the shortest path interpolation
        q1 = torch.where(torch.sum(q0*q1, dim=-1, keepdim=True) < 0, -q1, q1)

    A = q0.shape[:-1]
    B = steps.shape[: steps.dim() - len(A)]

    # Relative rotation
    rel_q = roma.quat_product(roma.quat_conjugation(q0), q1)                    # (A, 4)
    rel_rotvec = unitquat_to_full_rotvec(rel_q)                                 # (A, 3)
    # Relative rotations to apply
    rel_rotvecs = (                                                             # (B, A, 3)
        steps.unsqueeze(dim=-1)                                                 # (B, A, 1)
        * rel_rotvec.reshape((1,) * len(B) + rel_rotvec.shape)                  # (<|B| no. of 1>, A, 3)
    )
    rots = roma.rotvec_to_unitquat(                                             # (B, A, 4)
        rel_rotvecs.reshape(-1, 3)
    ).reshape(*rel_rotvecs.shape[:-1], 4)
    interpolated_q = roma.quat_product(                                         # (B, A, 4)
        q0.reshape((1,) * len(B) + q0.shape)                                    # (<|B| no. of 1>, A, 4)
          .repeat(B + (1,) * q0.dim()),                                         # (B, A, 4)
        rots                                                                    # (B, A, 4)
    )
    return interpolated_q
