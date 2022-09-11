import torch
import geoopt


def tanh(x):
    return x.clamp(-15, 15).tanh()

def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)

def tan_k(x: torch.Tensor, k: torch.Tensor):
    k_sqrt = geoopt.utils.sabs(k).sqrt()
    if k.dim() == 0 or k.size(0) == 1:
        scaled_x = x * k_sqrt
        return k_sqrt.reciprocal() * tanh(scaled_x)
    else:
        scaled_x = x * k_sqrt.view(-1,1)
        return k_sqrt.reciprocal().view(-1,1) * tanh(scaled_x)

def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sqrt = geoopt.utils.sabs(k).sqrt()
    if k.dim() == 0 or k.size(0) == 1:
        scaled_x = x * k_sqrt
        return k_sqrt.reciprocal() * artanh(scaled_x)
    else:
        scaled_x = x * k_sqrt.view(-1,1)
        return k_sqrt.reciprocal().view(-1,1) * artanh(scaled_x)

def expmap0(u: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _expmap0(u, k, dim=dim)

def _expmap0(u: torch.Tensor, k: torch.Tensor, dim: int = -1):
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    gamma_1 = tan_k(u_norm, k) * (u / u_norm)
    return gamma_1

def logmap0(y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _logmap0(y, k, dim=dim)

def _logmap0(y: torch.Tensor, k, dim: int = -1):
    y_norm = y.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    return (y / y_norm) * artan_k(y_norm, k)

def mobius_matvec(m: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _mobius_matvec(m, x, k, dim=dim)

def _mobius_matvec(m: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Mobius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    if dim != -1 or m.dim() == 2:
        mx = torch.tensordot(x, m, ([dim], [1]))
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = tan_k(mx_norm / x_norm * artan_k(x_norm, k), k) * (mx / mx_norm)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.bool)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res
    
def mobius_add(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _mobius_add(x, y, k, dim=dim)

def _mobius_add(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    if k.dim() == 0 or k.size(0) == 1:
        num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
        denom = 1 - 2 * k * xy + k ** 2 * x2 * y2    
    else:
        num = (1 - 2 * k.view(-1,1) * xy - k.view(-1,1) * y2) * x + (1 + k.view(-1,1) * x2) * y
        denom = 1 - 2 * k.view(-1,1) * xy + k.view(-1,1) ** 2 * x2 * y2
    return num / denom.clamp_min(1e-15)
    
def mobius_pointwise_mul(w: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _mobius_pointwise_mul(w, x, k, dim=dim)

def _mobius_pointwise_mul(w: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    wx = w * x
    wx_norm = wx.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = tan_k(wx_norm / x_norm * artan_k(x_norm, k), k) * (wx / wx_norm)
    zero = torch.zeros((), dtype=res_c.dtype, device=res_c.device)
    cond = wx.isclose(zero).prod(dim=dim, keepdim=True, dtype=torch.bool)
    res = torch.where(cond, zero, res_c)
    return res

def project(x: torch.Tensor, *, k: torch.Tensor, dim=-1, eps=-1):
    return _project(x, k, dim, eps)

def _project(x, k, dim: int = -1, eps: float = -1.0):
    if eps < 0:
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
    if k.dim() == 0 or k.size(0) == 1:
        maxnorm = (1 - eps) / (geoopt.utils.sabs(k) ** 0.5)
    else:
        maxnorm = (1 - eps) / (geoopt.utils.sabs(k).view(-1,1) ** 0.5)
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)
    