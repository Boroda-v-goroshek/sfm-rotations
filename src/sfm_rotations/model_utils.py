import torch
import torch.nn.functional as F

def skew_symmetric(v):
    """
    v: [..., 3]
    returns: [..., 3, 3]
    """
    x, y, z = v.unbind(dim=-1)

    zero = torch.zeros_like(x)

    return torch.stack([
        zero, -z, y,
        z, zero, -x,
        -y, x, zero
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)

def se3_exp(xi):
    """
    xi: [..., 6], порядок [rho, phi]
    returns: [..., 4, 4]
    """
    rho = xi[..., :3]
    phi = xi[..., 3:]

    theta2 = (phi * phi).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-12))

    # Коэффициенты SO(3) exponential map.
    A_regular = torch.sin(theta) / theta
    B_regular = (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12)

    # Taylor expansion для малых углов.
    A_small = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
    B_small = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0

    small = theta2 < 1e-6

    A = torch.where(small, A_small, A_regular)[..., None]
    B = torch.where(small, B_small, B_regular)[..., None]

    phi_hat = skew_symmetric(phi)
    phi_hat2 = phi_hat @ phi_hat

    eye = torch.eye(
        3, device=xi.device, dtype=xi.dtype
    ).expand(*xi.shape[:-1], 3, 3)

    R = eye + A * phi_hat + B * phi_hat2

    # Left Jacobian SO(3).
    t = R @ rho.unsqueeze(-1)

    T = torch.zeros(
        *xi.shape[:-1], 4, 4,
        device=xi.device,
        dtype=xi.dtype
    )

    T[..., :3, :3] = R
    T[..., :3, 3] = t.squeeze(-1)
    T[..., 3, 3] = 1.0

    return T

def make_se3(R, t):
    """
    R: [..., 3, 3] — rotation matrix
    t: [..., 3]    — translation vector

    Returns:
        T: [..., 4, 4]
    """
    T = torch.zeros(
        *R.shape[:-2], 4, 4,
        device=R.device,
        dtype=R.dtype
    )

    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0

    return T

def rotation_6d_to_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert the 6D rotation representation to rotation matrices.

    Uses Gram-Schmidt orthogonalization (Zhou et al., 2019).

    Parameters
    ----------
    x:
        6D rotations [B, 6].

    eps:
        Epsilon for vector normalization.

    Returns
    -------
    rotation matrices [B, 3, 3]

    """
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)
