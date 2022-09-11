"""Hyperboloid manifold."""

"""
    Hyperboloid manifold class.
    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K
    c = 1 / K is the hyperbolic curvature. 
    k = -c
"""


import torch

eps = {torch.float32: 1e-7, torch.float64: 1e-15}
min_norm = 1e-15
max_norm = 1e6

def minkowski_dot(x, y, keepdim=True):
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    res = torch.nan_to_num(torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0])
    if keepdim:
        res = res.view(res.shape + (1,))
    assert res.isfinite().all()
    return res

def minkowski_norm(u, keepdim=True):
    assert torch.isfinite(u).all()
    dot = minkowski_dot(u, u, keepdim=keepdim)
    assert torch.isfinite(dot).all()
    ret = torch.sqrt(torch.clamp(dot, min = eps[u.dtype]))
    assert ret.isfinite().all()
    return ret

def proj(x, k):
    assert torch.isfinite(x).all()
    assert k.isfinite().all()
    if k.size()[0] != 1:
        K = 1. / -k.view(-1,1)
    else:
        K = 1. / -k
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_sqnorm = torch.nan_to_num(torch.sum(y * y, dim=-1, keepdim=True))
    ret = torch.cat((torch.sqrt(torch.clamp(K + y_sqnorm, min=eps[x.dtype]).nan_to_num()), y), -1)
    assert ret.isfinite().all()
    return ret

def proj_tan(u, x, k):
    assert torch.isfinite(u).all()
    assert torch.isfinite(x).all()
    assert k.isfinite().all()
    if k.size()[0] != 1:
        K = 1. / -k.view(-1,1)
    else:
        K = 1. / -k
    d = x.size(-1) - 1
    ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=-1, keepdim=True)
    l_v = torch.nan_to_num(ux / torch.clamp(x.narrow(-1, 0, 1), min=eps[x.dtype]))
    ret = torch.cat((l_v, u.narrow(-1, 1, d)), -1)
    assert ret.isfinite().all()
    return ret

def expmap(u, x, k):
    assert torch.isfinite(u).all()
    assert torch.isfinite(x).all()
    if k.size()[0] != 1:
        K = 1. / -k.view(-1,1)
    else:
        K = 1. / -k
    sqrtK = K ** 0.5
    assert sqrtK.isfinite().all()
    normu = minkowski_norm(u)
    normu = torch.clamp(normu, max=max_norm)
    assert torch.isfinite(minkowski_norm(u)).all()
    assert torch.isfinite(normu).all()
    theta = - normu / sqrtK    
    theta = torch.clamp(theta, min=eps[normu.dtype], max=10.0)
    assert torch.isfinite(theta).all()
    result = torch.cosh(theta) * x + torch.sinh(theta) * u / theta
    assert torch.isfinite(result).all()
    return proj(result, k)
    
def expmap0(u, k):
    assert torch.isfinite(u).all()
    if k.size()[0] != 1:
        K = 1. / -k.view(-1,1)
    else:
        K = 1. / -k
    sqrtK = K ** 0.5
    assert sqrtK.isfinite().all()
    d = u.size(-1) - 1
    x = u.narrow(-1, 1, d)
    if x.size(0) == d:
        x = x.view(-1, d)
    x_norm = torch.clamp(torch.norm(x, p=2, dim=-1, keepdim=True), min=eps[x.dtype]).nan_to_num()
    theta = torch.clamp(x_norm / sqrtK, max=10.0)
#    if not theta.isfinite().all():
#        print(x_norm)
#        print(sqrtK)
    assert theta.isfinite().all()
    l_v = torch.nan_to_num(sqrtK * torch.cosh(theta))
    r_v = torch.nan_to_num(sqrtK * torch.sinh(theta) * x / x_norm)
#    if not l_v.isfinite().all() or not r_v.isfinite().all():
#        print("l_v=", l_v.isfinite().all())
#        print("r_v=", r_v.isfinite().all())
    assert l_v.isfinite().all()
    assert r_v.isfinite().all()
    res = torch.cat((l_v, r_v), -1)
    return proj(res, k)

def logmap0(x, k):
    assert torch.isfinite(x).all()
    if k.size()[0] != 1:
        K = 1. / -k.view(-1,1)
    else:
        K = 1. / -k
    sqrtK = K ** 0.5
    assert sqrtK.isfinite().all()
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_norm = torch.clamp(torch.norm(y, p=2, dim=-1, keepdim=True), min=eps[y.dtype]).nan_to_num()
    assert y_norm.isfinite().all()
    theta = torch.clamp(x.narrow(-1, 0, 1) / sqrtK, min=1.0 + eps[x.dtype], max = max_norm)
    assert theta.isfinite().all()
    assert torch.acosh(theta).isfinite().all()
    l_v = torch.zeros_like(x.narrow(-1, 0, 1))
    r_v = torch.nan_to_num(sqrtK * torch.acosh(theta) * y / y_norm)
#    if not r_v.isfinite().all():
#        print("r_v=", r_v.isfinite().all())
    assert r_v.isfinite().all()
    return torch.cat((l_v, r_v), -1)

def ptransp0(x, u, k):
    assert x.isfinite().all()
    assert u.isfinite().all()
    if k.size()[0] != 1:
        K = 1. / -k.view(-1,1)
    else:
        K = 1. / -k
    sqrtK = K ** 0.5
    assert sqrtK.isfinite().all()
    x0 = x.narrow(-1, 0, 1)
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_norm = torch.clamp(torch.norm(y, p=2, dim=-1, keepdim=True), min = eps[y.dtype]).nan_to_num()
    y_normalized = y / y_norm
    r_y = torch.nan_to_num((sqrtK - x0) * y_normalized)
    assert torch.isfinite(y_norm).all()
    assert torch.isfinite(y_normalized).all()
    assert torch.isfinite(r_y).all()
    v = torch.cat((- y_norm, r_y), -1)
    if not v.isfinite().all():
        print(sqrtK)
        print(x0)
        print(sqrtK - x0)
        print(k)
        print(v)
    assert v.isfinite().all()
    alpha = torch.sum(y_normalized * u.narrow(-1, 1, d), dim=-1, keepdim=True) / sqrtK
    assert torch.isfinite(alpha).all()
    res = u - alpha * v
    return proj_tan(res, x, k)

def mobius_add(x, y, k):
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert k.isfinite().all()
    u = logmap0(y, k)
    v = ptransp0(x, u, k)
    return expmap(v, x, k)

def mobius_matvec(m, x, k):
    assert torch.isfinite(m).all()
    assert torch.isfinite(x).all()
    assert k.isfinite().all()
    u = logmap0(x, k)
    #print(u.size())
    #print(m.transpose(-1,-2).size())
    mu = torch.matmul(u, m.transpose(-1,-2))
    return expmap0(mu, k)

def to_poincare(x, k):
    assert torch.isfinite(x).all()
    assert k.isfinite().all()
    K = 1. / -k.view(-1,1)
    sqrtK = K ** 0.5
    d = x.size(-1) - 1
    return sqrtK * x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + sqrtK)

#def sqdist(x, y, k):
#    K = 1. / -k
#    prod = minkowski_dot(x, y)
#    theta = torch.clamp(-prod / K, min=1.0 + eps[x.dtype])
#    sqdist = K * torch.acosh(theta) ** 2
#    # clamp distance to avoid nans in Fermi-Dirac decoder
#    ret = torch.clamp(sqdist, max=50.0)
#    return ret
#
#def logmap(x, y, k):
#    K = 1. / -k
#    xy = torch.clamp(minkowski_dot(x, y) + K, max=-eps[x.dtype]) - K
#    u = y + xy * x * -k
#    normu = minkowski_norm(u)
#    normu = torch.clamp(normu, min=min_norm)
#    dist = sqdist(x, y, k) ** 0.5
#    result = dist * u / normu
#    return proj_tan(result, x, k)
#
#def ptransp(x, y, u, k):
#    logxy = logmap(x, y, k)
#    logyx = logmap(y, x, k)
#    sqdist = torch.clamp(sqdist(x, y, k), min=min_norm)
#    alpha = minkowski_dot(logxy, u) / sqdist
#    res = u - alpha * (logxy + logyx)
#    return proj_tan(res, y, k)
#
#def proj_tan0(u, k):
#    narrowed = u.narrow(-1, 0, 1)
#    vals = torch.zeros_like(u)
#    vals[:, 0:1] = narrowed
#    return u - vals
#
