# This file is part of MOFTy.
#
# MOFTy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 3 as published by the Free
# Software Foundation.
#
# MOFTy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# MOFTy. If not, see http://www.gnu.org/licenses/
#
# Copyright(C) 2026 Maximilian Neumann, Philipp Arras

import nifty.cl as ift
import numpy as np
import numpy as np


def myassert(cond, msg):
    if not cond:
        raise RuntimeError(msg)


class MatrixMultiplyJac(ift.LinearOperator):
    def __init__(self, pos, target):
        myassert(isinstance(pos.domain, ift.MultiDomain), "bad domain")
        myassert(isinstance(target, ift.DomainTuple), "bad domain")
        self._target = target
        self._domain = pos.domain
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._pos = pos.val
        myassert(len(self._domain) == 2, "bad num of keys")
        self._k0, self._k1 = self._domain.keys()

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        a, b = self._pos[self._k0], self._pos[self._k1]
        if mode == self.TIMES:
            da, db = x[self._k0], x[self._k1]
            res = a @ db + da @ b
        elif mode == self.ADJOINT_TIMES:
            res = {self._k1: np.transpose(a) @ x, self._k0: x @ np.transpose(b)}
        return ift.makeField(self._tgt(mode), res)


class MatrixMultiply(ift.Operator):
    def __init__(self, dom0, key0, dom1, key1):
        dom0 = ift.DomainTuple.make(dom0)
        dom1 = ift.DomainTuple.make(dom1)
        myassert(len(dom0) == 2, "bad domain")
        myassert(len(dom1) == 2, "bad domain")
        myassert(dom0[1] == dom1[0], "bad domain")
        myassert(len(dom0[0].shape) == 1, "bad domain")
        myassert(len(dom0[1].shape) == 1, "bad domain")
        myassert(len(dom1[0].shape) == 1, "bad domain")
        myassert(len(dom1[1].shape) == 1, "bad domain")
        self._key0 = str(key0)
        self._key1 = str(key1)
        self._domain = ift.makeDomain({self._key0: dom0, self._key1: dom1})
        self._target = ift.makeDomain((dom0[0], dom1[1]))

    def apply(self, x):
        self._check_input(x)
        if not ift.is_linearization(x):  # no Jacobian needed
            fac0 = x[self._key0].val
            fac1 = x[self._key1].val
            return ift.makeField(self._target, np.matmul(fac0, fac1))
        else:  # with Jacobian
            fac0 = x[self._key0].val.val
            fac1 = x[self._key1].val.val
            val = ift.makeField(self._target, fac0 @ fac1)
            jac = MatrixMultiplyJac(x.val, self._target)
            return x.new(val, jac)


class MaskOperator(ift.LinearOperator):
    def __init__(self, mask):
        assert isinstance(mask, ift.Field)
        assert mask.dtype == bool
        self._domain = ift.DomainTuple.make(mask.domain)
        self._mask = mask.val
        self._not_mask = ~mask.val
        self._target = ift.makeDomain(ift.UnstructuredDomain(self._mask.sum()))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def _device_preparation(self, x):
        self._mask = self._mask.at(x.device_id)
        self._not_mask = self._not_mask.at(x.device_id)

    def apply(self, x, mode):
        self._check_input(x, mode)
        self._device_preparation(x)
        if mode == self.TIMES:
            return ift.Field(self.target, x.val[self._mask])
        elif mode == self.ADJOINT_TIMES:
            res = np.empty_like(x.val, shape=self.domain.shape, dtype=x.dtype)
            res[self._mask] = x.val
            res[self._not_mask] = 0
            return ift.Field(self.domain, res)
        raise RuntimeError()


class FactorwiseLinearInterpolator(ift.LinearOperator):
    """
    A LinearInterpolator that broadcasts over a leading factor dimension.
    """

    def __init__(self, domain, sampling_points):
        if not isinstance(domain, (tuple, list)) or len(domain) != 2:
            raise TypeError("Input domain must be a tuple or list of length 2.")

        dom_factors_orig, dom_spatial_orig = domain

        if not isinstance(dom_factors_orig, ift.UnstructuredDomain):
            raise TypeError(
                "The first part of the domain must be an UnstructuredDomain."
            )
        if not isinstance(dom_spatial_orig, ift.RGSpace):
            raise TypeError("The second part of the domain must be an RGSpace.")

        self._domain = ift.DomainTuple.make(domain)
        self._interpolator = ift.LinearInterpolator(
            dom_spatial_orig, sampling_points=sampling_points
        )
        n_points = sampling_points.shape[1]
        dom_samples = ift.UnstructuredDomain(n_points)
        self._target = ift.DomainTuple.make((dom_factors_orig, dom_samples))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            input_vals = x.val
            output_vals = []
            dom_spatial = self.domain[1]
            for i in range(self.domain[0].shape[0]):
                spatial_slice_domain = ift.DomainTuple.make(dom_spatial)
                spatial_slice = ift.Field(spatial_slice_domain, input_vals[i])
                interpolated_slice = self._interpolator(spatial_slice)
                output_vals.append(interpolated_slice.val)
            final_val = np.stack(output_vals, axis=0)
            return ift.Field(self.target, final_val)
        else:
            input_vals = x.val
            output_vals = []
            dom_samples = self.target[1]
            for i in range(self.target[0].shape[0]):
                sample_slice_domain = ift.DomainTuple.make(dom_samples)
                sample_slice = ift.Field(sample_slice_domain, input_vals[i])
                adjoint_slice = self._interpolator.adjoint(sample_slice)
                output_vals.append(adjoint_slice.val)
            final_val = np.stack(output_vals, axis=0)
            return ift.Field(self.domain, final_val)
