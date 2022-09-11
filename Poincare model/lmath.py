import torch

eps = {torch.float32: 1e-7, torch.float64: 1e-15}
min_norm = 1e-15
max_norm = 1e6

def minkowski_dot(x, y, keepdim=True):
    res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
    if keepdim:
        res = res.view(res.shape + (1,))
    return res

def minkowski_norm(u, keepdim=True):
    dot = minkowski_dot(u, u, keepdim=keepdim)
    return torch.sqrt(torch.clamp(dot, min=eps[u.dtype]))

def sqdist(x, y, k):
    prod = minkowski_dot(x, y)
    theta = torch.clamp(-prod / k, min=1.0 + eps[x.dtype])
    sqdist = k * torch.acosh(theta) ** 2
    # clamp distance to avoid nans in Fermi-Dirac decoder
    return torch.clamp(sqdist, max=50.0)

def proj(x, k):
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2 
    mask = torch.ones_like(x)
    mask[:, 0] = 0
    vals = torch.zeros_like(x)
    vals[:, 0:1] = torch.sqrt(torch.clamp(k + y_sqnorm, min=eps[x.dtype]))
    return vals + mask * x

def proj_tan(u, x, k):
    d = x.size(1) - 1
    ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
    mask = torch.ones_like(u)
    mask[:, 0] = 0
    vals = torch.zeros_like(u)
    vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=eps[x.dtype])
    return vals + mask * u

def proj_tan0(u, k):
    narrowed = u.narrow(-1, 0, 1)
    vals = torch.zeros_like(u)
    vals[:, 0:1] = narrowed
    return u - vals

def expmap(u, x, k):
    sqrtk = k ** 0.5
    normu = minkowski_norm(u)
    normu = torch.clamp(normu, max=max_norm)
    theta = normu / sqrtk
    theta = torch.clamp(theta, min=min_norm)
    result = torch.cosh(theta) * x + torch.sinh(theta) * u / theta
    return proj(result, k)
    
def logmap(x, y, k):
    xy = torch.clamp(minkowski_dot(x, y) + k, max=-eps[x.dtype]) - k
    u = y + xy * x / k
    normu = minkowski_norm(u)
    normu = torch.clamp(normu, min=min_norm)
    dist = sqdist(x, y, k) ** 0.5
    result = dist * u / normu
    return proj_tan(result, x, k)

def expmap0(u, k, dim=-1):
    nomin = _norm(u, keepdim=True, dim=dim)
    l_v = torch.cosh(nomin / torch.sqrt(k)) * torch.sqrt(k)
    r_v = torch.sqrt(k) * torch.sinh(nomin / torch.sqrt(k)) * u / nomin
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    return p

def logmap0(x, k):
    sqrtk = k ** 0.5
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d).view(-1, d)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
    y_norm = torch.clamp(y_norm, min=min_norm)
    res = torch.zeros_like(x)
    theta = torch.clamp(x[:, 0:1] / sqrtk, min=1.0 + eps[x.dtype])
    res[:, 1:] = sqrtk * torch.acosh(theta) * y / y_norm
    return res

def mobius_add(x, y, k):
    u = logmap0(y, k)
    v = ptransp0(x, u, k)
    return expmap(v, x, k)

def mobius_matvec(m, x, k):
    u = logmap0(x, k)
    mu = u @ m.transpose(-1, -2)
    return expmap0(mu, k)

def ptransp(x, y, u, k):
    logxy = logmap(x, y, k)
    logyx = logmap(y, x, k)
    sqdist = torch.clamp(sqdist(x, y, k), min=min_norm)
    alpha = minkowski_dot(logxy, u) / sqdist
    res = u - alpha * (logxy + logyx)
    return proj_tan(res, y, k)

def ptransp0(x, u, k):
    sqrtk = k ** 0.5
    x0 = x.narrow(-1, 0, 1)
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=min_norm)
    y_normalized = y / y_norm
    v = torch.ones_like(x)
    v[:, 0:1] = - y_norm 
    v[:, 1:] = (sqrtk - x0) * y_normalized
    alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtk
    res = u - alpha * v
    return proj_tan(res, x, k)

def to_poincare(x, k):
    sqrtk = k ** 0.5
    d = x.size(-1) - 1
    return sqrtk * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtk)
    
