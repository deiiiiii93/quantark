"""Reusable transition kernels and explicit corrections for partial cells."""
from collections import OrderedDict
import numpy as np

from quantark.util.exceptions import NumericalError
from .basis import gauss, lagrange
from .gaussian import affine_moments, readout
from .direct import apply_direct
from .fft import apply_fft


class ByteCache:
    def __init__(self, limit):
        self.limit, self.bytes = limit, 0
        self.entries = OrderedDict()

    def get(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        if value.nbytes > self.limit:
            return
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes
        while self.bytes + value.nbytes > self.limit:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.nbytes
        self.entries[key] = value
        self.bytes += value.nbytes


#: Gauss-Legendre points per standard-normal panel of the local kernel, and the panel width (sd): a cell polynomial
#: times the density over 2 sd is integrated to round-off (S2's "Eq" form, robust at every s/h; the tests check it
#: against adaptive quadrature to 1e-12 for s/h from 1e-4 to 1.9)
_LOCAL_POINTS = 16
_LOCAL_PANEL_SD = 2.0


class Operator:
    def __init__(self, mesh, params, local_variance=0.0):
        """``local_variance``: intervals of smaller variance are integrated by the exact local kernel (``kernel``);
        0 (every context the mesh sizing fits) keeps today's kernels for every interval."""
        self.mesh, self.params = mesh, params
        self.local_variance = float(local_variance)
        self.cache = ByteCache(params.max_cache_bytes)
        self.backends_used = set()

    def kernel(self, mean, variance, lo=0.0, hi=1.0):
        # Exact moment keys: distinct intervals are never silently rounded.
        key = (float(mean), float(variance), float(lo), float(hi))
        cached = self.cache.get(key)
        if cached is not None:
            return cached, key
        if variance < self.local_variance:
            kernel = self._local_kernel(float(mean), float(variance), float(lo), float(hi))
            self.cache.put(key, kernel)
            return kernel, key
        mesh = self.mesh
        band = max(
            1,
            int(np.ceil((abs(mean) + self.params.tail_sd * np.sqrt(variance)) / mesh.h))
            + 1,
        )
        n_sub = 2 * mesh.order + 8
        required = (2 * band + 1) * n_sub * mesh.order * 8 * 4
        if required > self.params.max_work_bytes:
            raise NumericalError(f"QUAD V2 kernel workspace requires {required} bytes")
        gx, gw = gauss(n_sub)
        t, wt = (hi - lo) * gx / 2 + (hi + lo) / 2, (hi - lo) * gw / 2
        z = (
            np.arange(-band, band + 1)[:, None, None] + t[None, :, None] - mesh.theta
        ) * mesh.h - mean
        density = np.exp(-z * z / (2 * variance)) / np.sqrt(2 * np.pi * variance)
        kernel = mesh.h * np.einsum(
            "n,qn,dnr->dqr", wt, lagrange(mesh.theta, t), density, optimize=True
        )
        self.cache.put(key, kernel)
        return kernel, key

    def _local_kernel(self, mean, variance, lo, hi):
        """The cell kernel of an interval the cells do not resolve (sub-resolution run S4).

        ``kernel[band + d, q, r]`` is the integral of source cell ``j``'s Lagrange polynomial ``L_q`` over its local
        coordinates ``[lo, hi]`` against the Gaussian density of the move from target node ``r`` of cell ``j - d``:
        in standard normal coordinates ``z``, ``t = theta_r - d + (mean + sd z) / h`` (offset ``d = j - i``), so the
        integral is ``int L_q(t(z)) phi(z) dz`` between the exact limits of ``[lo, hi]``, truncated at ``tail_sd`` as
        today's band. The limits cut fixed 2-sd panels, each integrated by 16-point Gauss-Legendre: exact to
        round-off for the degree-``order`` polynomial times the density at any s/h, where today's kernel samples the
        density on a cell grid that a step narrower than a cell aliases. The closed-form truncated moments (S2's
        "EaA") drift (4.6e-8 at s/h = 2) from the monomial expansion; this form does not.
        """
        mesh, params = self.mesh, self.params
        sd, h, theta = np.sqrt(variance), mesh.h, mesh.theta
        tail = float(params.tail_sd)
        band = max(1, int(np.ceil((abs(mean) + tail * sd) / h)) + 1)
        edges = np.r_[np.arange(-tail, tail, _LOCAL_PANEL_SD), tail]
        d = np.arange(-band, band + 1)[:, None]
        za = np.clip(((d + lo - theta) * h - mean) / sd, -tail, tail)              # (D, R)
        zb = np.clip(((d + hi - theta) * h - mean) / sd, -tail, tail)
        a = np.maximum(za[..., None], edges[:-1])                                   # (D, R, P)
        b = np.maximum(a, np.minimum(zb[..., None], edges[1:]))
        live = np.flatnonzero((b > a).any(axis=(1, 2)))
        kernel = np.zeros((2 * band + 1, mesh.order, mesh.order))
        if not len(live):
            return kernel
        a, b = a[live], b[live]
        required = a.size * _LOCAL_POINTS * (mesh.order + 3) * 8
        if required > params.max_work_bytes:
            raise NumericalError(f"QUAD V2 local kernel workspace requires {required} bytes")
        gx, gw = gauss(_LOCAL_POINTS)
        half, mid = (b - a) / 2, (b + a) / 2
        z = mid[..., None] + half[..., None] * gx                                  # (L, R, P, G)
        weights = half[..., None] * gw * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        t = theta[None, :, None, None] - d[live][..., None, None] + (mean + sd * z) / h
        kernel[live] = np.einsum("qlrpg,lrpg->lqr", lagrange(mesh.theta, t), weights, optimize=True)
        return kernel

    def integrate(self, function, mean, variance, discount):
        mesh = self.mesh
        # Deterministic compositions may shift a source mesh. Use the same
        # cell-wise Gaussian readout, without nodal projection of its jumps.
        if any(
            s.shift != 0 or s.grid.mesh != mesh
            for p in function.pieces
            for s in p.sources
        ):
            result = readout(
                function, mesh.nodes.ravel(), mean, variance, discount, self.params
            )
            return result[0].reshape(function.channels, mesh.cells, mesh.order)
        out = np.zeros((function.channels, mesh.cells, mesh.order))
        merged = np.zeros_like(out)
        has_sources = False
        for p in function.pieces:
            if np.any(p.cash):
                cash = affine_moments(mesh.nodes, mean, variance, p.lo, p.hi, 0)[0]
                out += p.cash[:, None, None] * cash
            if np.any(p.asset):
                asset = affine_moments(mesh.nodes, mean, variance, p.lo, p.hi, 1)[0]
                out += p.asset[:, None, None] * asset
            if not p.sources:
                continue
            has_sources = True
            ua = max(0.0, (p.lo - mesh.lo) / mesh.h)
            ub = min(float(mesh.cells), (p.hi - mesh.lo) / mesh.h)
            if ua >= ub:
                continue
            start, stop = int(np.ceil(ua)), int(np.floor(ub))
            for src in p.sources:
                merged[:, start:stop] += src.weight * src.grid.values[:, start:stop]
            partial = []
            if int(np.floor(ua)) == int(np.floor(ub)):
                partial.append((int(np.floor(ua)), ua % 1, ub % 1))
            else:
                if ua < start:
                    partial.append((int(np.floor(ua)), ua % 1, 1.0))
                if stop < ub:
                    partial.append((stop, 0.0, ub - stop))
            for cell, a, b in partial:
                if b <= a or cell >= mesh.cells:
                    continue
                kernel, _ = self.kernel(mean, variance, a, b)
                band = (len(kernel) - 1) // 2
                i0, i1 = max(0, cell - band), min(mesh.cells, cell + band + 1)
                idx = cell - np.arange(i0, i1) + band
                for src in p.sources:
                    out[:, i0:i1] += src.weight * np.einsum(
                        "cq,nqr->cnr", src.grid.values[:, cell], kernel[idx]
                    )
        if has_sources:
            kernel, key = self.kernel(mean, variance)
            backend = self.params.backend
            if backend == "auto":
                backend = "fft" if len(kernel) * mesh.cells > 10_000 else "direct"
            self.backends_used.add(backend)
            if backend == "fft":
                out += apply_fft(
                    kernel, merged, self.params.max_work_bytes, self.cache, key
                )
            else:
                out += apply_direct(kernel, merged, self.params.max_work_bytes)
        return discount * out
