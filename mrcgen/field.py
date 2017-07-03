#! /usr/bin/env python

from sympy import *

######################################################################


def Parameter(s):
    return Symbol("ctx->params.%s" % (s))

######################################################################


def toScalar(s):
    if isinstance(s, ScalarBase):
        return s

    return ScalarTerm(s)
    # if type(s) == int or type(s) == float:
    #     return ScalarTerm(s)

    # if isinstance(s, core.Basic):
    #     return ScalarTerm(s)

    # print "type = ", type(s)
    # raise TypeError

######################################################################


class Field(object):
    def __str__(self):
        return str(self.at((0, 0, 0)))


######################################################################
# Scalars

class ScalarBase(Field):
    def __init__(self):
        self.terms = []

    def __call__(self, d0, d1, d2):
        return ScalarShift(self, (d0, d1, d2))

    def __neg__(s):
        return ScalarNeg(s)

    def __add__(s1, s2):
        return ScalarAdd(s1, s2)

    def __radd__(s1, s2):
        return ScalarAdd(s2, s1)

    def __sub__(s1, s2):
        return ScalarSub(s1, s2)

    def __rsub__(s1, s2):
        return ScalarSub(s2, s1)

    def __mul__(s1, s2):
        if isinstance(s2, VectorBase):
            return VectorSMul(s1, s2)

        if isinstance(s2, TensorBase):
            return TensorSMul(s1, s2)

        return ScalarMul(s1, s2)

    def __rmul__(s1, s2):
        return ScalarMul(s2, s1)

    def __div__(s1, s2):
        return ScalarDiv(s1, s2)

    def __rdiv__(s1, s2):
        return ScalarDiv(s2, s1)


class ScalarShift(ScalarBase):
    def __init__(self, s, d):
        self.terms = [s]
        self.d = d

    def at(self, i):
        d = self.d
        return self.terms[0].at((i[0] + d[0],
                                 i[1] + d[1],
                                 i[2] + d[2]))


class ScalarNeg(ScalarBase):
    def __init__(self, s):
        self.terms = [toScalar(s)]

    def at(self, i):
        return -self.terms[0].at(i)


class ScalarBinOp(ScalarBase):
    def __init__(self, s1, s2):
        self.terms = [toScalar(s1), toScalar(s2)]


class ScalarAdd(ScalarBinOp):
    def at(self, i):
        return self.terms[0].at(i) + self.terms[1].at(i)

    def face(self, f):
        return self.terms[0].face(f) + self.terms[1].face(f)


class ScalarSub(ScalarBinOp):
    def at(self, i):
        return self.terms[0].at(i) - self.terms[1].at(i)


class ScalarMul(ScalarBinOp):
    def at(self, i):
        return self.terms[0].at(i) * self.terms[1].at(i)

    def face(self, f):
        return self.terms[0].face(f) * self.terms[1].face(f)


# FIXME: Hack -- topological terms shouldn't be handled in the same
# way as physical scalar fields, but I don't know how many other places
# have hacks to work around the incorrect behavior of ScalarBase.__mul__,
# so I need to just put this in for now.
# When the codegenerator is refactored we need
# to handle this more consistently.
class ScalarTopoMul(ScalarMul):
    def face(self, m):
        # FIXME: This assumes terms[1] is the topological quantity
        if m == 0:
            return .5 * (self.terms[0](-1, 0, 0) * self.terms[1](-1, 0, 0) +
                         self.terms[0](0, 0, 0) * self.terms[1](0, 0, 0))
        if m == 1:
            return .5 * (self.terms[0](0, -1, 0) * self.terms[1](0, -1, 0) +
                         self.terms[0](0, 0, 0) * self.terms[1](0, 0, 0))

        if m == 2:
            return .5 * (self.terms[0](0, 0, -1) * self.terms[1](0, 0, -1) +
                         self.terms[0](0, 0, 0) * self.terms[1](0, 0, 0))


class ScalarDiv(ScalarBinOp):
    def at(self, i):
        return self.terms[0].at(i) / self.terms[1].at(i)

    def face(self, m):
        return self.terms[0].face(m) / self.terms[1].face(m)


class ScalarTerm(ScalarBase):
    def __init__(self, t):
        self.terms = []
        self.t = t

    def at(self, i):
        return self.t

    def face(self, m):
        return self


class ScalarField(ScalarBase):
    def __init__(self, str, m=0):
        self.terms = []
        self.str = str
        self.m = m

    def at(self, i):
        return FLD(self.str, self.m, self).at(i)

    def face(self, m):
        return FLD(self.str, self.m, self).face(m)


class ScalarFieldFromVectorField(ScalarBase):
    def __init__(self, v, m):
        self.v = v
        self.terms = []
        self.str = v.str
        self.m = m

    def at(self, i):
        return FLD(self.str, self.v.m + self.m, self).at(i)

    def face(self, m):
        return FLD(self.str, self.v.m + self.m, self).face(m)

    def __getattr__(self, attr):
        if attr == "bc":
            return self.v.bc[self.m]

        raise AttributeError


class ScalarAvg(ScalarBase):
    def face(self, m):
        if m == 0:
            return .5 * (self(-1, 0, 0) + self(0, 0, 0))
        if m == 1:
            return .5 * (self(0, -1, 0) + self(0, 0, 0))
        if m == 2:
            return .5 * (self(0, 0, -1) + self(0, 0, 0))


# FIXME, definitely a hack, and not really even helping
class sp_zero(ScalarBase):
    def at(self, i):
        if i[0] == -1:
            return Symbol("sp_zero")
        else:
            return 1.


######################################################################
# Vectors

class VectorBase(Field):
    def __call__(self, d0, d1, d2):
        class VectorShift(VectorBase):
            def __init__(self, v, d):
                self.v, self.d = v, d

            def __getitem__(self, m):
                d = self.d
                return self.v[m](d[0], d[1], d[2])

        return VectorShift(self, (d0, d1, d2))

    def __neg__(v):
        return VectorNeg(v)

    def __add__(v1, v2):
        return VectorAdd(v1, v2)

    def __sub__(v1, v2):
        return VectorSub(v1, v2)

    def __mul__(v, s):
        return VectorSMul(s, v)

    def __rmul__(v, s):
        return VectorSMul(s, v)


class VectorNeg(VectorBase):
    def __init__(self, v):
        self.v = v

    def __getitem__(self, m):
        return -self.v[m]


class VectorBinOp(VectorBase):
    def __init__(self, v1, v2):
        self.v1, self.v2 = v1, v2


class VectorAdd(VectorBinOp):
    def __getitem__(self, m):
        return self.v1[m] + self.v2[m]


class VectorSub(VectorBinOp):
    def __getitem__(self, m):
        return self.v1[m] - self.v2[m]


class VectorSMul(VectorBase):
    def __init__(self, s, v):
        self.s, self.v = toScalar(s), v

    def __getitem__(self, m):
        return self.s * self.v[m]

    def face(self, m):
        return self.s.face(m) * self.v.face(m)


class VectorSMulZIP(VectorSMul):
    def face(self, d):
        if d == 0:
            return .5 * (self.s(-1, 0, 0) * self.v(0, 0, 0) +
                         self.s(0, 0, 0) * self.v(-1, 0, 0))
        if d == 1:
            return .5 * (self.s(0, -1, 0) * self.v(0, 0, 0) +
                         self.s(0, 0, 0) * self.v(0, -1, 0))
        if d == 2:
            return .5 * (self.s(0, 0, -1) * self.v(0, 0, 0) +
                         self.s(0, 0, 0) * self.v(0, 0, -1))

# We need something to handle V=P/RHO correctly. We probably shouldn't
# make this the default return from __div__ until we're sure that the jacobian
# will also be handled correctly.
# (JAC and RHO being the same type is problematic and not really accurate)


class VectorSDiv(VectorBase):
    '''
    A form of V/S that yields the face
    (V/S)_{1/2} = 0.5 * (V_{0}/S_{0} + V_{1}/S_{1})
    '''
    def __init__(self, v, s):
        self.v, self.s = v, s

    def __getitem__(self, m):
        return self.v[m] / self.s

    def face(self, m):
        class Face(VectorBase):
            def __init__(self, m, v):
                self.m, self.v = m, v

            def __getitem__(self, i):
                m = self.m
                s, v = self.v.s, self.v.v

                if not m == i:
                    raise Error
                if i == 0:
                    v_i = v[i] / JAC()
                    fac = JAC().face(m)
                else:
                    v_i = v[i]
                    fac = 1

                if m == 0:
                    return .5 * (v_i(0, 0, 0) / s(0, 0, 0) +
                                 v_i(-1, 0, 0) / s(0, 0, 0)) * fac
                if m == 1:
                    return .5 * (v[i](0, 0, 0) / s(0, 0, 0) +
                                 v[i](0, -1, 0) / s(0, -1, 0))
                if m == 2:
                    return .5 * (v[i](0, 0, 0) / s(0, 0, 0) +
                                 v[i](0, 0, -1) / s(0, 0, -1))

        return Face(m, self)


class VectorField(VectorBase):
    def __init__(self, str, m=0):
        self.str, self.m = str, m

    def __getitem__(self, m):
        return ScalarFieldFromVectorField(self, m)

    def face(self, f):
        if f == 0:
            return .5 * (self(-1, 0, 0) + self(0, 0, 0))
        if f == 1:
            return .5 * (self(0, -1, 0) + self(0, 0, 0))
        if f == 2:
            return .5 * (self(0, 0, -1) + self(0, 0, 0))


class VectorScl(VectorBase):
    def __init__(self, s):
        self.s = s

    def __getitem__(self, m):
        return self.s[m]


class VectorSV(VectorBase):
    def __init__(self, s, v):
        self.s, self.v = s, v

    def __getitem__(self, m):
        return self.s * self.v[m]

    def face(self, m):
        class Face(VectorBase):
            def __init__(self, m, v):
                self.m, self.v = m, v

            def __getitem__(self, i):
                m = self.m
                s, v = self.v.s, self.v.v
                if m == 0:
                    return .5 * (s(-1, 0, 0) * v[i](0, 0, 0) +
                                 s(0, 0, 0) * v[i](-1, 0, 0))
                if m == 1:
                    return .5 * (s(0, -1, 0) * v[i](0, 0, 0) +
                                 s(0, 0, 0) * v[i](0, -1, 0))
                if m == 2:
                    return .5 * (s(0, 0, -1) * v[i](0, 0, 0) +
                                 s(0, 0, 0) * v[i](0, 0, -1))

        return Face(m, self)


class VectorSV_cyl(VectorSV):
    def face(self, m):
        class Face(VectorBase):
            def __init__(self, m, v):
                self.m, self.v = m, v

            def __getitem__(self, i):
                m = self.m
                s, v = self.v.s, self.v.v

                if not m == i:
                    raise Error
                if i == 0:
                    v_i = v[i] / JAC()
                    fac = JAC().face(m)
                else:
                    v_i = v[i]
                    fac = 1

                if m == 0:
                    return .5 * (s(-1, 0, 0) * v_i(0, 0, 0) +
                                 s(0, 0, 0) * v_i(-1, 0, 0)) * fac * sp_zero()
                if m == 1:
                    return .5 * (s(0, -1, 0) * v[i](0, 0, 0) +
                                 s(0, 0, 0) * v[i](0, -1, 0))
                if m == 2:
                    return .5 * (s(0, 0, -1) * v[i](0, 0, 0) +
                                 s(0, 0, 0) * v[i](0, 0, -1))

        return Face(m, self)


class ScalarSimplify(ScalarBase):
    def __init__(self, s):
        self.s = s

    def at(self, i):
        return simplify(self.s.at(i))


class VectorSimplify(VectorBase):
    def __init__(self, v):
        self.v = v

    def __getitem__(self, m):
        return ScalarSimplify(self.v[m])


def simplify_block(term):
    if isinstance(term, VectorBase):
        return VectorSimplify(term)
    elif isinstance(term, ScalarBase):
        return ScalarSimplify(term)
    else:
        raise TypeError("Cannot simplify term of type %s", type(term))


######################################################################
# Tensors

class TensorBase(Field):
    def __call__(self, d0, d1, d2):
        class TensorShift(VectorBase):
            def __init__(self, v, d):
                self.v, self.d = v, d

            def __getitem__(self, ij):
                d = self.d
                return self.v[ij](d[0], d[1], d[2])

        return TensorShift(self, (d0, d1, d2))

    def __neg__(t):
        return TensorNeg(t)

    def __add__(t1, t2):
        return TensorAdd(t1, t2)

    def __sub__(t1, t2):
        return TensorSub(t1, t2)

    def __mul__(t, s):
        return TensorSMul(s, t)

    def __rmul__(t, s):
        return TensorSMul(s, t)


class TensorNeg(TensorBase):
    def __init__(self, t):
        self.t = t

    def __getitem__(self, ij):
        return -self.t[ij]

    def face(self, m):
        return -self.t.face(m)


class TensorBinOp(TensorBase):
    def __init__(self, t1, t2):
        self.t1, self.t2 = t1, t2


class TensorAdd(TensorBinOp):
    def __getitem__(self, ij):
        return self.t1[ij] + self.t2[ij]

    def face(self, m):
        return self.t1.face(m) + self.t2.face(m)


class TensorSub(TensorBinOp):
    def __getitem__(self, ij):
        return self.t1[ij] - self.t2[ij]

    def face(self, m):
        return self.t1.face(m) - self.t2.face(m)


class TensorSMul(TensorBase):
    def __init__(self, s, t):
        self.s, self.t = toScalar(s), t

    def __getitem__(self, ij):
        return self.s * self.t[ij]

    def face(self, m):
        return self.s.face(m) * self.t.face(m)


class TensorAvg(TensorBase):
    def face(self, m):
        if m == 0:
            return .5 * (self(-1, 0, 0) + self(0, 0, 0))
        if m == 1:
            return .5 * (self(0, -1, 0) + self(0, 0, 0))
        if m == 2:
            return .5 * (self(0, 0, -1) + self(0, 0, 0))


class TensorGUU(TensorAvg):
    def __getitem__(self, ij):
        return GUU(ij)


class TensorGLL(TensorAvg):
    def __getitem__(self, ij):
        return GLL(ij)


class TensorTSS(TensorBase):
    def __init__(self, t, s1, s2):
        self.t, self.s1, self.s2 = t, s1, s2

    def __getitem__(self, ij):
        return self.t[ij] * self.s1 * self.s2

    def face(self, m):
        t, s1, s2 = self.t, self.s1, self.s2
        if m == 0:
            return (t.face(m) *
                    .5 * (s1(-1, 0, 0) * s2(0, 0, 0) +
                          s1(0, 0, 0) * s2(-1, 0, 0)))
        if m == 1:
            return (t.face(m) *
                    .5 * (s1(0, -1, 0) * s2(0, 0, 0) +
                          s1(0, 0, 0) * s2(0, -1, 0)))
        if m == 2:
            return (t.face(m) *
                    .5 * (s1(0, 0, -1) * s2(0, 0, 0) +
                          s1(0, 0, 0) * s2(0, 0, -1)))


class TensorST(TensorBase):
    def __init__(self, s1, t2):
        self.s1, self.t2 = s1, t2

    def __getitem__(self, ij):
        return self.s1 * self.t2[ij]

    def face(self, m):
        class Face(TensorBase):
            def __init__(self, m, t):
                self.m, self.t = m, t

            def __getitem__(self, ij):
                s1, t2 = self.t.s1, self.t.t2

                if m == 0:
                    return .5 * (s1(-1, 0, 0) * t2[ij](0, 0, 0) +
                                 s1(0, 0, 0) * t2[ij](-1, 0, 0))
                if m == 1:
                    return .5 * (s1(0, -1, 0) * t2[ij](0, 0, 0) +
                                 s1(0, 0, 0) * t2[ij](0, -1, 0))
                if m == 2:
                    return .5 * (s1(0, 0, -1) * t2[ij](0, 0, 0) +
                                 s1(0, 0, 0) * t2[ij](0, 0, -1))

        return Face(m, self)


class COV(VectorBase):
    def __init__(self, v):
        self.v = v

    def __getitem__(self, m):
        v = self.v
        gll = TensorGLL()

        return (gll[(m, 0)] * v[0] +
                gll[(m, 1)] * v[1] +
                gll[(m, 2)] * v[2])


class CNV(VectorBase):
    def __init__(self, v):
        self.v = v

    def __getitem__(self, m):
        v = self.v
        guu = TensorGUU()

        return (guu[(m, 0)] * v[0] +
                guu[(m, 1)] * v[1] +
                guu[(m, 2)] * v[2])


class TensorBpress(TensorBase):
    def __init__(self, t, B):
        self.t, self.B = t, B

    def __getitem__(self, ij):
        t, B = self.t, self.B
        Bcov = COV(B)

        return t[ij] * (Bcov[0] * B[0] +
                        Bcov[1] * B[1] +
                        Bcov[2] * B[2])

    def face(self, m):
        t, B = self.t, self.B
        Bcov = COV(B)

        if m == 0:
            return (t.face(m) *
                    .5 * (Bcov[0](-1, 0, 0) * B[0](0, 0, 0) +
                          Bcov[0](0, 0, 0) * B[0](-1, 0, 0) +
                          Bcov[1](-1, 0, 0) * B[1](0, 0, 0) +
                          Bcov[1](0, 0, 0) * B[1](-1, 0, 0) +
                          Bcov[2](-1, 0, 0) * B[2](0, 0, 0) +
                          Bcov[2](0, 0, 0) * B[2](-1, 0, 0)))
        if m == 1:
            return (t.face(m) *
                    .5 * (Bcov[0](0, -1, 0) * B[0](0, 0, 0) +
                          Bcov[0](0, 0, 0) * B[0](0, -1, 0) +
                          Bcov[1](0, -1, 0) * B[1](0, 0, 0) +
                          Bcov[1](0, 0, 0) * B[1](0, -1, 0) +
                          Bcov[2](0, -1, 0) * B[2](0, 0, 0) +
                          Bcov[2](0, 0, 0) * B[2](0, -1, 0)))
        if m == 2:
            return (t.face(m) *
                    .5 * (Bcov[0](0, 0, -1) * B[0](0, 0, 0) +
                          Bcov[0](0, 0, 0) * B[0](0, 0, -1) +
                          Bcov[1](0, 0, -1) * B[1](0, 0, 0) +
                          Bcov[1](0, 0, 0) * B[1](0, 0, -1) +
                          Bcov[2](0, 0, -1) * B[2](0, 0, 0) +
                          Bcov[2](0, 0, 0) * B[2](0, 0, -1)))


class TensorVV(TensorBase):
    def __init__(self, v1, v2):
        self.v1, self.v2 = v1, v2

    def __getitem__(self, ij):
        i, j = ij
        return self.v1[i] * self.v2[j]

    def face(self, m):
        class Face(TensorBase):
            def __init__(self, m, t):
                self.m, self.t = m, t

            def __getitem__(self, ij):
                i, j = ij
                m, v1, v2 = self.m, self.t.v1, self.t.v2

                if m == 0:
                    return .5 * (v1[i](-1, 0, 0) * v2[j](0, 0, 0) +
                                 v1[i](0, 0, 0) * v2[j](-1, 0, 0))
                if m == 1:
                    return .5 * (v1[i](0, -1, 0) * v2[j](0, 0, 0) +
                                 v1[i](0, 0, 0) * v2[j](0, -1, 0))
                if m == 2:
                    return .5 * (v1[i](0, 0, -1) * v2[j](0, 0, 0) +
                                 v1[i](0, 0, 0) * v2[j](0, 0, -1))

        return Face(m, self)


class TensorVV_cyl(TensorVV):
    def face(self, m):
        class Face(TensorBase):
            def __init__(self, m, t):
                self.m, self.t = m, t

            def __getitem__(self, ij):
                i, j = ij
                m, v1, v2 = self.m, self.t.v1, self.t.v2
                if not i == m:
                    raise Error

                if m == 0:
                    if i == 0:
                        v1_i = v1[i] / J()
                        fac1 = J().face(m)
                    else:
                        v1_i = v1[i]
                        fac1 = 1

                    if j == 1:
                        v2_j = v2[j]
                        fac2 = 1
                    else:
                        v2_j = v2[j] / J()
                        fac2 = J().face(m)

                    return (.5 * (v1_i(-1, 0, 0) * v2_j(0, 0, 0) +
                                  v1_i(0, 0, 0) * v2_j(-1, 0, 0)) *
                            fac1 * fac2 * sp_zero())
                if m == 1:
                    return .5 * (v1[i](0, -1, 0) * v2[j](0, 0, 0) +
                                 v1[i](0, 0, 0) * v2[j](0, -1, 0))
                if m == 2:
                    return .5 * (v1[i](0, 0, -1) * v2[j](0, 0, 0) +
                                 v1[i](0, 0, 0) * v2[j](0, 0, -1))

        return Face(m, self)


######################################################################
# divergence
def ScalarDivg(v):
    return ((v.face(0)[0](1, 0, 0) - v.face(0)[0](0, 0, 0)) /
            (XI0f(1) - XI0f(0)) +
            (v.face(1)[1](0, 1, 0) - v.face(1)[1](0, 0, 0)) /
            (XI1f(1) - XI1f(0)) +
            (v.face(2)[2](0, 0, 1) - v.face(2)[2](0, 0, 0)) /
            (XI2f(1) - XI2f(0)))


class VectorDivg(VectorBase):
    def __init__(self, t):
        self.t = t

    def __getitem__(self, i):
        tJ = 1. / JAC() * self.t

        return (simplify_block((tJ.face(0)[(0, i)](1, 0, 0) -
                                tJ.face(0)[(0, i)](0, 0, 0)) /
                (XI0f(1) - XI0f(0))) +
                (tJ.face(1)[(1, i)](0, 1, 0) -
                    tJ.face(1)[(1, i)](0, 0, 0)) / (XI1f(1) - XI1f(0)) +
                simplify_block((tJ.face(2)[(2, i)](0, 0, 1) -
                               tJ.face(2)[(2, i)](0, 0, 0)) /
                (XI2f(1) - XI2f(0))) +

                tJ[(0, 0)] * GAM(i, 0, 0) +
                tJ[(0, 1)] * GAM(i, 0, 1) +
                tJ[(0, 2)] * GAM(i, 0, 2) +
                tJ[(1, 0)] * GAM(i, 1, 0) +
                tJ[(1, 1)] * GAM(i, 1, 1) +
                tJ[(1, 2)] * GAM(i, 1, 2) +
                tJ[(2, 0)] * GAM(i, 2, 0) +
                tJ[(2, 1)] * GAM(i, 2, 1) +
                tJ[(2, 2)] * GAM(i, 2, 2))


class DivgAlt(VectorBase):
    def __init__(self, t):
        self.t = t

    def __getitem__(self, i):
        t = self.t

        if i != 1:
            return VectorDivg(self.t)[i]

        return (1. / JAC() *
                ((t.face(0)[(0, i)](1, 0, 0) - t.face(0)[(0, i)](0, 0, 0)) /
                 (XI0f(1) - XI0f(0)) +
                 (t.face(1)[(1, i)](0, 1, 0) - t.face(1)[(1, i)](0, 0, 0)) /
                 (XI1f(1) - XI1f(0)) +
                 (t.face(2)[(2, i)](0, 0, 1) - t.face(2)[(2, i)](0, 0, 0)) /
                 (XI2f(1) - XI2f(0)) +

                 - t[(0, i)] * (GAM(0, 0, 0) + GAM(1, 0, 1) + GAM(2, 0, 2))
                 - t[(1, i)] * (GAM(0, 1, 0) + GAM(1, 1, 1) + GAM(2, 1, 2))
                 - t[(2, i)] * (GAM(0, 2, 0) + GAM(1, 2, 1) + GAM(2, 2, 2))

                 + t[(0, 0)] * GAM(i, 0, 0) + t[(0, 1)] * GAM(i, 0, 1) + t[(0, 2)] * GAM(i, 0, 2)
                 + t[(1, 0)] * GAM(i, 1, 0) + t[(1, 1)] * GAM(i, 1, 1) + t[(1, 2)] * GAM(i, 1, 2)
                 + t[(2, 0)] * GAM(i, 2, 0) + t[(2, 1)] * GAM(i, 2, 1) + t[(2, 2)] * GAM(i, 2, 2)))


def Divg(x):
    if isinstance(x, TensorBase):
        return VectorDivg(x)

    if isinstance(x, VectorBase):
        return ScalarDivg(x)

    raise TypeError


def DivGrad(s):
    guu = TensorGUU()
    # FIXME 3d
    g00, g01, g11, g22 = guu[(0, 0)], guu[(0, 1)], guu[(1, 1)], guu[(2, 2)]

    return (1 / (XI0f(1) - XI0f(0)) * (.5 * (g00(1, 0, 0) + g00(0, 0, 0)) *
                                       (s(1, 0, 0) - s(0, 0, 0)) /
                                       (XI0(1) - XI0(0)) -
                                       .5 * (g00(0, 0, 0) + g00(-1, 0, 0)) *
                                       (s(0, 0, 0) - s(-1, 0, 0)) /
                                       (XI0(0) - XI0(-1))) +
            1 / (XI1f(1) - XI1f(0)) * (.5 * (g11(0, 1, 0) + g11(0, 0, 0)) *
                                       (s(0, 1, 0) - s(0, 0, 0)) /
                                       (XI1(1) - XI1(0)) -
                                       .5 * (g11(0, 0, 0) + g11(0, -1, 0)) *
                                       (s(0, 0, 0) - s(0, -1, 0)) /
                                       (XI1(0) - XI1(-1))) +
            1 / (XI2f(1) - XI2f(0)) * (.5 * (g22(0, 0, 1) + g22(0, 0, 0)) *
                                       (s(0, 0, 1) - s(0, 0, 0)) /
                                       (XI2(1) - XI2(0)) -
                                       .5 * (g22(0, 0, 0) + g22(0, 0, -1)) *
                                       (s(0, 0, 0) - s(0, 0, -1)) /
                                       (XI2(0) - XI2(-1))) +
            .5 / ((XI0f(1) - XI0f(0)) * (XI1f(1) - XI1f(0))) *
            (.25 * (g01(0, 0, 0) + g01(1, 0, 0) +
                    g01(0, 1, 0) + g01(1, 1, 0)) *
             (s(1, 1, 0) - s(0, 0, 0)) +
             .25 * (g01(0, 0, 0) + g01(-1, 0, 0) +
                    g01(0, -1, 0) + g01(-1, -1, 0)) *
             (s(-1, -1, 0) - s(0, 0, 0)) -
             .25 * (g01(0, 0, 0) + g01(1, 0, 0) +
                    g01(0, -1, 0) + g01(1, -1, 0)) *
             (s(1, -1, 0) - s(0, 0, 0)) -
             .25 * (g01(0, 0, 0) + g01(-1, 0, 0) +
                    g01(0, 1, 0) + g01(-1, 1, 0)) *
             (s(-1, 1, 0) - s(0, 0, 0))))


def DivGradNum(s):
    # guu = TensorGUU()

    print "WARNING, can't be right..."
    return (((s(1, 0, 0) - s(0, 0, 0)) - (s(0, 0, 0) - s(-1, 0, 0))) +
            ((s(0, 1, 0) - s(0, 0, 0)) - (s(0, 0, 0) - s(0, -1, 0))) +
            ((s(0, 0, 1) - s(0, 0, 0)) - (s(0, 0, 0) - s(0, 0, -1))))


def Dc_l(l, s):
    if l == 0:
        return (s(1, 0, 0) - s(-1, 0, 0)) / (XI0(1) - XI0(-1))
    if l == 1:
        return (s(0, 1, 0) - s(0, -1, 0)) / (XI1(1) - XI1(-1))
    if l == 2:
        return (s(0, 0, 1) - s(0, 0, -1)) / (XI2(1) - XI2(-1))


class GradSV(TensorBase):
    def __init__(self, s, v):
        self.s, self.v = s, v

    def __getitem__(self, ij):
        s, v = self.s, self.v
        k, m = ij
        rv = 0.
        guu = TensorGUU()
        for l in range(3):
            rv += (s * guu[(k, l)] *
                   (Dc_l(l, v[m]) -
                    (v[m] * gam(0, l, 0) +
                     v[m] * gam(1, l, 1) +
                     v[m] * gam(2, l, 2)) +
                    (v[0] * gam(m, 0, l) +
                     v[1] * gam(m, 1, l) +
                     v[2] * gam(m, 2, l))))
        return rv

    def face(self, f):
        class Face(TensorBase):
            def __init__(self, f, p):
                self.f, self.p = f, p

            def __getitem__(self, ij):
                f, s, v = self.f, self.p.s, self.p.v
                k, m = ij
                if k != f:
                    raise Error
                guu = TensorGUU()

                def D_l(l, vm, f):
                    if not isinstance(vm, ScalarBase):
                        raise TypeError
                    if l == f:
                        if l == 0:
                            return ((vm(0, 0, 0) - vm(-1, 0, 0)) /
                                    (XI0f(0) - XI0f(-1)))
                        if l == 1:
                            return ((vm(0, 0, 0) - vm(0, -1, 0)) /
                                    (XI1f(0) - XI1f(-1)))
                        if l == 2:
                            return ((vm(0, 0, 0) - vm(0, 0, -1)) /
                                    (XI2f(0) - XI2f(-1)))

                    else:
                        if f == 0:
                            return .5 * (Dc_l(l, vm)(-1, 0, 0) +
                                         Dc_l(l, vm)(0, 0, 0))
                        if f == 1:
                            return .5 * (Dc_l(l, vm)(0, -1, 0) +
                                         Dc_l(l, vm)(0, 0, 0))
                        if f == 2:
                            return .5 * (Dc_l(l, vm)(0, 0, -1) +
                                         Dc_l(l, vm)(0, 0, 0))

                rv = 0.
                for l in range(3):
                    rv += (s.face(f) * guu[(k, l)].face(f) *
                           (D_l(l, v[m], f) -
                            (v[m] * gam(0, l, 0) +
                             v[m] * gam(1, l, 1) +
                             v[m] * gam(2, l, 2)).face(f) +
                            (v[0] * gam(m, 0, l) +
                             v[1] * gam(m, 1, l) +
                             v[2] * gam(m, 2, l)).face(f)))

                return rv

        return Face(f, self)


def Dc_lcyl(l, v, m):
    if l == 0:
        if m in [0, 2]:
            return (JAC() * ((v[m] / JAC())(1, 0, 0) -
                             (v[m] / JAC())(-1, 0, 0)) / (XI0(1) - XI0(-1)) +
                    (JAC()(1, 0, 0) - JAC()(-1, 0, 0)) / (XI0(1) - XI0(-1)) *
                    v[m] / JAC())

        else:
            return (v[m](1, 0, 0) - v[m](-1, 0, 0)) / (XI0(1) - XI0(-1))

    if l == 1:
        return (v[m](0, 1, 0) - v[m](0, -1, 0)) / (XI1(1) - XI1(-1))
    if l == 2:
        return (v[m](0, 0, 1) - v[m](0, 0, -1)) / (XI2(1) - XI2(-1))


class GradSV_cyl(GradSV):
    def __getitem__(self, ij):
        s, v = self.s, self.v
        k, m = ij
        rv = 0.
        guu = TensorGUU()
        for l in range(3):
            rv += (s * guu[(k, l)] *
                   (Dc_lcyl(l, v, m) -
                    (v[m] * gam(0, l, 0) +
                     v[m] * gam(1, l, 1) +
                     v[m] * gam(2, l, 2)) +
                    (v[0] * gam(m, 0, l) +
                     v[1] * gam(m, 1, l) +
                     v[2] * gam(m, 2, l))))
        return rv

    def face(self, f):
        class Face(TensorBase):
            def __init__(self, f, p):
                self.f, self.p = f, p

            def __getitem__(self, ij):
                f, s, v = self.f, self.p.s, self.p.v
                k, m = ij
                if k != f:
                    raise Error
                guu = TensorGUU()

                def D_l(l, v, m, f):
                    if not isinstance(v[m], ScalarBase):
                        raise TypeError
                    if l == f:
                        if l == 0:
                            # FIXME -- it certainly doesn't work this way...
                            # Actually, I'm pretty sure it sort of does...
                            if m in [0, 2]:
                                return (JAC().face(l) *
                                        ((v[m] / JAC())(0, 0, 0) -
                                            (v[m] / JAC())(-1, 0, 0)) /
                                        (XI0(0) - XI0(-1)))
                            else:
                                # we're basically just counting on
                                # the alt formulation
                                # being used here, otherwise
                                # this won't have the extra
                                # zero that we need
                                return ((v[m](0, 0, 0) - v[m](-1, 0, 0)) /
                                        (XI0(0) - XI0(-1)))

                        if l == 1:
                            return ((v[m](0, 0, 0) - v[m](0, -1, 0)) /
                                    (XI1(0) - XI1(-1)))
                        if l == 2:
                            return ((v[m](0, 0, 0) - v[m](0, 0, -1)) /
                                    (XI2(0) - XI2(-1)))

                    else:
                        if f == 0:
                            return .5 * (Dc_lcyl(l, v, m)(-1, 0, 0) +
                                         Dc_lcyl(l, v, m)(0, 0, 0))
                        if f == 1:
                            return .5 * (Dc_lcyl(l, v, m)(0, -1, 0) +
                                         Dc_lcyl(l, v, m)(0, 0, 0))
                        if f == 2:
                            return .5 * (Dc_lcyl(l, v, m)(0, 0, -1) +
                                         Dc_lcyl(l, v, m)(0, 0, 0))

                rv = 0.
                for l in range(3):
                    if l in [0, 2]:
                        v_m = v[m] / JAC()
                        fac = JAC().face(f)
                    else:
                        v_m = v[m]
                        fac = 1.0
                    rv += (s.face(f) * guu[(k, l)].face(f) *
                           (D_l(l, v, m, f) -
                            (v_m * gam(0, l, 0) +
                             v_m * gam(1, l, 1) +
                             v_m * gam(2, l, 2)).face(f) * fac +
                            ((v[0] / JAC()) *
                                gam(m, 0, l)).face(f) * JAC().face(f) +
                            (v[1] * gam(m, 1, l)).face(f) +
                            ((v[2] / JAC()) * gam(m, 2, l)).face(f) *
                            JAC().face(f)))

                    # This should be the correct version.
                    # Maybe we're not getting
                    # cancellations with the geometric terms?
                    # Causes significantly worse stability than alternative.
                    # (ScalarTopoMul(v_m,gam(0,l,0))+
                    #  ScalarTopoMul(v_m,gam(1,l,1))+
                    #  ScalarTopoMul(v_m,gam(2,l,2))).face(f) * fac +
                    # ScalarTopoMul(v[0]/JAC(),gam(m,0,l)).face(f)*JAC().face(f)+
                    # ScalarTopoMul(v[1],gam(m,1,l)).face(f)+
                    # ScalarTopoMul(v[2]/JAC(),gam(m,2,l)).face(f)*JAC().face(f)))

                return rv

        return Face(f, self)


class CROSS(VectorBase):
    """ calculates the covariant components of the cross product of
        v and B, given as contravariant components."""

    def __init__(self, v, B):
        self.v, self.B = v, B

    def __getitem__(self, m):
        v, B = self.v, self.B

        if m == 0:
            return 1. / JAC() * (v[1] * B[2] - v[2] * B[1])
        if m == 1:
            return 1. / JAC() * (v[2] * B[0] - v[0] * B[2])
        if m == 2:
            return 1. / JAC() * (v[0] * B[1] - v[1] * B[0])


class CURL(VectorBase):
    """ given covariant components of v, calculates contravariant components
        of the curl of v."""

    def __init__(self, v):
        if not isinstance(v, VectorBase):
            raise TypeError
        self.v = v

    def __getitem__(self, m):
        v = self.v

        if m == 0:
            return ((v[2](0, 1, 0) - v[2](0, -1, 0)) / (XI1(1) - XI1(-1)) -
                    (v[1](0, 0, 1) - v[1](0, 0, -1)) / (XI2(1) - XI2(-1)))
        if m == 1:
            return ((v[0](0, 0, 1) - v[0](0, 0, -1)) / (XI2(1) - XI2(-1)) -
                    (v[2](1, 0, 0) - v[2](-1, 0, 0)) / (XI0(1) - XI0(-1)))
        if m == 2:
            return ((v[1](1, 0, 0) - v[1](-1, 0, 0)) / (XI0(1) - XI0(-1)) -
                    (v[0](0, 1, 0) - v[0](0, -1, 0)) / (XI1(1) - XI1(-1)))


# FIXME: Don't use this with J..
# it would require massive BC hacks and still completely
# ruins the convergence of the Jacobian (for some reason).
class CURL_fd4(VectorBase):
    """ given covariant components of v, calculates contravariant components
          of the curl of v using a 4th order accurate finite-difference scheme.

    Note: using this to calculate a B field will not provide divB=0
    with the current discretizion of the divergence.
    It might with a different divergence."""

    def __init__(self, v):
        if not isinstance(v, VectorBase):
            raise TypeError
        self.v = v

    def __getitem__(self, m):
        v = self.v

        if m == 0:
            return (2. * (-1. / 12. * v[2](0, 2, 0) +
                          2. / 3. * v[2](0, 1, 0) -
                          2. / 3. * v[2](0, -1, 0) +
                          1. / 12. * v[2](0, -2, 0)) / (XI1(1) - XI1(-1)) -
                    2. * (-1. / 12. * v[1](0, 0, 2) +
                          2. / 3. * v[1](0, 0, 1) -
                          2. / 3. * v[1](0, 0, -1) +
                          1. / 12. * v[1](0, 0, -2)) / (XI2(1) - XI2(-1)))
        if m == 1:
            return (2. * (-1. / 12. * v[0](0, 0, 2) +
                          2. / 3. * v[0](0, 0, 1) -
                          2. / 3. * v[0](0, 0, -1) +
                          1. / 12. * v[0](0, 0, -2)) / (XI2(1) - XI2(-1)) -
                    2. * (-1. / 12. * v[2](2, 0, 0) +
                          2. / 3. * v[2](1, 0, 0) -
                          2. / 3. * v[2](-1, 0, 0) +
                          1. / 12. * v[2](-2, 0, 0)) / (XI0(1) - XI0(-1)))

        if m == 2:
            return (2. * (-1. / 12. * v[1](2, 0, 0) +
                          2. / 3. * v[1](1, 0, 0) -
                          2. / 3. * v[1](-1, 0, 0) +
                          1. / 12. * v[1](-2, 0, 0)) / (XI0(1) - XI0(-1)) -
                    2. * (-1. / 12. * v[0](0, 2, 0) +
                          2. / 3. * v[0](0, 1, 0) -
                          2. / 3. * v[0](0, -1, 0) +
                          1. / 12. * v[0](0, -2, 0)) / (XI1(1) - XI1(-1)))


class CURL_COMP(VectorBase):
    """ given covariant components of v, calculates contravariant components
        of the curl of v.
        Fixes a numerical issue, but introduces large error"""

    def __init__(self, v):
        if not isinstance(v, VectorBase):
            raise TypeError
        self.v = v

    def __getitem__(self, m):
        v = self.v

        if m == 0:
            return ((v[2](0, 1, 0) - v[2](0, -1, 0)) / (XI1(1) - XI1(-1)) -
                    (v[1](0, 0, 1) - v[1](0, 0, -1)) / (XI2(1) - XI2(-1)))
        if m == 1:
            return ((v[0](0, 0, 1) - v[0](0, 0, -1)) / (XI2(1) - XI2(-1)) -
                    (v[2](1, 0, 0) - v[2](-1, 0, 0)) / (XI0(1) - XI0(-1)))
        if m == 2:
            return ((XI0f(1) * .5 * (v[1](1, 0, 0) / XI0(1) +
                                     v[1](0, 0, 0) / XI0(0)) -
                     XI0f(0) * .5 * (v[1](-1, 0, 0) / XI0(-1) +
                                     v[1](0, 0, 0) / XI0(0))
                     ) / (XI0f(1) - XI0f(0)) -
                    (v[0](0, 1, 0) - v[0](0, -1, 0)) / (XI1(1) - XI1(-1)))


# FIXME, the whole cnv/cov stuff can be done much nicer...
def CURL_cnv(vcon):
    """ given contravariant components of v, calculates contravariant components
        of the curl of v."""

    return CURL(COV(vcon))


######################################################################
# trafo-specific
class TrafoBase:
    class CRD:
        pass

    class JAC(ScalarBase):
        pass

    class GUU(ScalarBase):
        pass

    class GLL(ScalarBase):
        pass

    class GAM(ScalarBase):
        pass


class Trafo3d(TrafoBase):
    class CRD:
        def __init__(self, m):
            self.m = m

        def __getitem__(self, idx):
            return Symbol("XI%d(crds,j%c,patch)" % (self.m, 'x' + self.m))

    class JAC(ScalarAvg):
        def at(self, idx):
            return Symbol("JAC(jx+%d,jy+%d,jz+%d)" % idx)

    class FLD(ScalarAvg):
        def __init__(self, str, m, fld):
            self.str, self.m, self.fld = str, m, fld

        def at(self, idx):
            return Symbol("MRC_D5(%s,%d, jx+%d,jy+%d,jz+%d,patch)" %
                          (self.str, self.m, idx[0], idx[1], idx[2]))

    class GAM(ScalarBase):
        def __init__(self, i, j, k):
            self.i, self.j, self.k = i, j, k

        def at(self, idx):
            return Symbol("GAM(trafo,%d,%d,%d, jx+%d,jy+%d,jz+%d,patch)"
                          % (self.i, self.j, self.k, idx[0], idx[1], idx[2]))

    class GUU(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            return Symbol("GUU(trafo,%d,%d, jx+%d,jy+%d,jz+%d,patch)"
                          % (ij[0], ij[1], idx[0], idx[1], idx[2]))

    class GLL(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            return Symbol("GLL(trafo,%d,%d, jx+%d,jy+%d,jz+%d,patch)"
                          % (ij[0], ij[1], idx[0], idx[1], idx[2]))


class F3(Symbol):
    pass
#     def __init__(self, str, s, m):
#         Symbol.__init__(self, str)
#         print "F3 init ", s, m, i
#         Symbol.__init__(self, "F3(%s,%d, jx+%d,jy+%d,jz+%d)" %
#                         (s, m, i[0], i[1], i[2]))


class Trafo2d(TrafoBase):
    class CRD(ScalarAvg):
        def __init__(self, m):
            self.m = m

        def at(self, idx):
            m = self.m
            return Symbol("XI%d(crds,j%s+%d,patch)"
                          % (m, ['x', 'y', 'z'][m], idx[m]))

        def __call__(self, d):
            class CRDShift(ScalarBase):
                def __init__(self, s, d):
                    self.s, self.d = s, d

                def at(self, i):
                    d = self.d
                    return self.s.at((i[0] + d[0], i[1] + d[1], i[2] + d[2]))

            dd = [0, 0, 0]
            dd[self.m] = d
            return CRDShift(self, dd)

    class JAC(ScalarAvg):
        def at(self, idx):
            return Symbol("JAC(trafo,jx+%d,jy+%d,0,patch)" % (idx[0], idx[1]))

    class FLD(ScalarAvg):
        def __init__(self, str, m, fld):
            self.str, self.m, self.fld = str, m, fld
            self.terms = []

        def at(self, idx):
            f3 = F3("MRC_D5(%s,%d, jx+%d,jy+%d,0,patch)" %
                    (self.str, self.m, idx[0], idx[1]))
            # FIXME, ugly hack...
            f3.str = self.str
            f3.m = self.m
            f3.idx = (idx[0], idx[1], 0)
            f3.fld = self.fld
            return f3

    class GAM(ScalarAvg):
        def __init__(self, i, j, k):
            self.ijk = i, j, k

        def at(self, idx):
            ijk = self.ijk
            if True or ijk in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                               (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
                return Symbol("__GAM(trafo,%d, jx+%d,jy+%d,0,patch)"
                              % (((ijk[0] * 3) + ijk[1]) * 3 + ijk[2],
                                 idx[0], idx[1]))
            else:
                return 0.

    class GUU(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            return Symbol("trafo,GUU(%d,%d, jx+%d,jy+%d,0,patch)"
                          % (ij[0], ij[1], idx[0], idx[1]))

    class GLL(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            return Symbol("GLL(trafo,%d,%d, jx+%d,jy+%d,0,patch)"
                          % (ij[0], ij[1], idx[0], idx[1]))


class Trafo2dSinusoidal(Trafo2d):
    # FIXME, should be a tensor...
    class GUU(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            if (ij in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]):
                return Symbol("GUU(trafo,%d,%d, jx+%d,jy+%d,0,patch)"
                              % (ij[0], ij[1], idx[0], idx[1]))
            else:
                return 0.

    class GLL(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            if (ij in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]):
                return Symbol("GLL(trafo,%d,%d, jx+%d,jy+%d,0,patch)"
                              % (ij[0], ij[1], idx[0], idx[1]))
            else:
                return 0.


class Trafo2dCylindrical(Trafo2d):
    class CRD(ScalarAvg):
        def __init__(self, m):
            self.m = m

        def at(self, idx):
            m = self.m
            if m < 2:
                return Symbol("XI%d(crds,j%s+%d,patch)" %
                              (m, ['x', 'y', 'z'][m], idx[m]))
            else:
                return Parameter("twist") * Symbol("XI1(crds,%d,patch)" %
                                                   (idx[2]))

        def __call__(self, d):
            class CRDShift(ScalarBase):
                def __init__(self, s, d):
                    self.s, self.d = s, d

                def at(self, i):
                    d = self.d
                    return self.s.at((i[0] + d[0],
                                      i[1] + d[1],
                                      i[2] + d[2]))

            dd = [0, 0, 0]
            dd[self.m] = d
            return CRDShift(self, dd)

    class JAC(ScalarAvg):
        def at(self, idx):
            return Symbol("XI0(crds,jx+%d,patch)" %
                          (idx[0]))

    class FLD(ScalarAvg):
        def __init__(self, str, m, fld):
            self.str, self.m, self.fld = str, m, fld
            self.terms = []

        def at(self, idx):
            f3 = F3("MRC_D5(%s,%d, jx+%d,jy+%d,0,patch)" %
                    (self.str, self.m, idx[0], idx[1] - idx[2]))
            # FIXME, ugly hack...
            f3.str = self.str
            f3.m = self.m
            f3.idx = (idx[0], idx[1] - idx[2], 0)
            f3.fld = self.fld
            return f3

    class GUU(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            if ij == (0, 0):
                return Symbol("XI0(crds,jx+%d,patch)" % (idx[0]))
            if ij == (1, 1):
                return 1. / Symbol("XI0(crds,jx+%d,patch)" % (idx[0]))
            if ij == (2, 2):
                return Symbol("XI0(crds,jx+%d,patch)" % (idx[0]))

            return 0.

    class GLL(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            if ij == (0, 0):
                return 1. / Symbol("XI0(crds,jx+%d,patch)" % (idx[0]))
            if ij == (1, 1):
                return Symbol("XI0(crds,jx+%d,patch)" % (idx[0]))
            if ij == (2, 2):
                return 1. / Symbol("XI0(crds,jx+%d,patch)" % (idx[0]))

            return 0.

    class GAM(ScalarAvg):
        def __init__(self, i, j, k):
            self.ijk = (i, j, k)

        def at(self, idx):
            ijk = self.ijk
            if (ijk in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]):
                return Symbol("GAM(trafo,%d,%d,%d, jx+%d,jy+%d,0,patch)" %
                              (ijk[0], ijk[1], ijk[2],
                               idx[0], idx[1] - idx[2]))
            else:
                return 0.


class Trafo2dCartesian(Trafo2d):
    # FIXME, all of these are constant (but not nec. == 1),
    # can be optimized
    class JAC(ScalarAvg):
        def at(self, idx):
            return 1.

    class GUU(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            if (ij in [(0, 0), (1, 1), (2, 2)]):
                return 1.
            else:
                return 0.

    class GLL(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            if (ij in [(0, 0), (1, 1), (2, 2)]):
                return 1.
            else:
                return 0.

    class GAM(ScalarAvg):
        def __init__(self, i, j, k):
            self.ijk = (i, j, k)

        def at(self, idx):
            return 0.


class Trafo3dCartesian(Trafo3d):
    # FIXME, all of these are constant (but not nec. == 1),
    # can be optimized
    class JAC(ScalarAvg):
        def at(self, idx):
            return 1.

    class GUU(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            if (ij in [(0, 0), (1, 1), (2, 2)]):
                return 1.
            else:
                return 0.

    class GLL(ScalarAvg):
        def __init__(self, ij):
            self.ij = ij

        def at(self, idx):
            ij = self.ij
            if (ij in [(0, 0), (1, 1), (2, 2)]):
                return 1.
            else:
                return 0.

    class GAM(ScalarAvg):
        def __init__(self, i, j, k):
            self.ijk = (i, j, k)

        def at(self, idx):
            return 0.


global JAC, GAM, CRD, GUU, GLL, FLD, XI0, XI1, XI2, XI0f, XI1f, XI2f


def J():
    return JAC()


def guu():
    return TensorGUU()


def gam(i, j, k):
    return GAM(i, j, k)


def XI0f(jx):
    return .5 * (XI0(jx - 1) + XI0(jx))


def XI1f(jy):
    return .5 * (XI1(jy - 1) + XI1(jy))


def XI2f(jz):
    return .5 * (XI2(jz - 1) + XI2(jz))


def setTrafo(trafo):
    global JAC, GAM, CRD, GUU, GLL, FLD, XI0, XI1, XI2, XI0f, XI1f, XI2f
    JAC = trafo.JAC
    GAM = trafo.GAM
    CRD = trafo.CRD
    GUU = trafo.GUU
    GLL = trafo.GLL
    FLD = trafo.FLD

    XI0 = CRD(0)
    XI1 = CRD(1)
    XI2 = CRD(2)


######################################################################
def postorder_traversal(node):
    """ Do a postorder traversal of a tree.

    This generator recursively yields nodes that it has visited in a postorder
    fashion. That is, it descends through the tree depth-first to yield all of
    a node's children's postorder traversal before yielding the node itself.

    """
    for term in node.terms:
        for subtree in postorder_traversal(term):
            yield subtree
        yield term
