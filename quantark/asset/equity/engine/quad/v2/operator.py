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


class Operator:
    def __init__(self, mesh, params):
        self.mesh, self.params = mesh, params
        self.cache = ByteCache(params.max_cache_bytes)
        self.backends_used = set()

    def kernel(self, mean, variance, lo=0.0, hi=1.0):
        # Exact moment keys: distinct intervals are never silently rounded.
        key = (float(mean), float(variance), float(lo), float(hi))
        cached = self.cache.get(key)
        if cached is not None:
            return cached, key
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
