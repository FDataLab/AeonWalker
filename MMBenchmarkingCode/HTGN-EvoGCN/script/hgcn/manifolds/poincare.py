import torch
from script.hgcn.manifolds.base import Manifold
from script.hgcn.utils.math_utils import artanh, tanh


class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        c = torch.as_tensor(c, device=p1.device)
        sqrt_c = c ** 0.5
        dist_c = artanh(sqrt_c * self.mobius_add(-p1, p2, c).norm(dim=-1))
        return (2 * dist_c / sqrt_c) ** 2

    def dist0(self, p1, c, keepdim=False):
        c = torch.as_tensor(c, device=p1.device)
        sqrt_c = c ** 0.5
        dist_c = artanh(sqrt_c * p1.norm(dim=-1, keepdim=keepdim))
        return 2 * dist_c / sqrt_c

    def _lambda_x(self, x, c):
        c = torch.as_tensor(c, device=x.device)
        x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
        return 2 / (1 - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        c = torch.as_tensor(c, device=p.device)
        return dp / self._lambda_x(p, c).pow(2)

    def proj(self, x, c):
        c = torch.as_tensor(c, device=x.device)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / c.sqrt()
        return torch.where(norm > maxnorm, x / norm * maxnorm, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        c = torch.as_tensor(c, device=u.device)
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        lambda_p = self._lambda_x(p, c)
        second_term = tanh(sqrt_c / 2 * lambda_p * u_norm) * u / (sqrt_c * u_norm)
        return self.mobius_add(p, second_term, c)

    def logmap(self, p1, p2, c):
        c = torch.as_tensor(c, device=p1.device)
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        c = torch.as_tensor(c, device=u.device)
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        return tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)

    def logmap0(self, p, c):
        c = torch.as_tensor(c, device=p.device)
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        return p * artanh(sqrt_c * p_norm) / (sqrt_c * p_norm)

    def mobius_add(self, x, y, c, dim=-1):
        c = torch.as_tensor(c, device=x.device)
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        c = torch.as_tensor(c, device=x.device)
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        return torch.where(cond, res_0, res_c)

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim=-1):
        c = torch.as_tensor(c, device=u.device)
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        c = torch.as_tensor(c, device=x.device)
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        c = torch.as_tensor(c, device=x.device)
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        c = torch.as_tensor(c, device=x.device)
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        c = torch.as_tensor(c, device=x.device)
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        c = torch.as_tensor(c, device=x.device)
        K = 1 / c
        sqrtK = K.sqrt()
        sqnorm = x.norm(p=2, dim=1, keepdim=True).pow(2)
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)
