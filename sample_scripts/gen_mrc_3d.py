#! /usr/bin/env python

import mrcgen.field
from mrcgen.field import *
from mrcgen.cgen import *

import sys

if len(sys.argv) < 2:
    raise ParseError

test_bc = False

if sys.argv[1] == "MRC_SINUSOIDAL":
    setTrafo(Trafo2dSinusoidal())
    gname = "sinusoidal"
elif sys.argv[1] == "MRC_CARTESIAN":
    setTrafo(Trafo2dCartesian())
    gname = "cartesian"
elif sys.argv[1] == "MRC_CYLINDRICAL":
    setTrafo(Trafo2dCylindrical())
    gname = "cylindrical"
elif sys.argv[1] == "MRC_BUTTERFLY":
    setTrafo(Trafo2d())
    gname = "butterfly"
else:
    raise ParseError

USE_HYPERRESISTIVITY = False

D = Parameter("D")
DT = Parameter("DT")
nu = Parameter("nu")
eta = Parameter("eta")
eta2 = Parameter("eta2")
gamma = Parameter("gam")
tau = Parameter("tau")
d_i = Parameter("d_i")
# print "WARNING: d_i = 0"

RHO = ScalarField("x", 0)
T = ScalarField("x", 1)
P = VectorField("x", 2)
B = VectorField("x", 5)

V = VectorSDiv(P, RHO)
# V = 1. / RHO * P

XI0 = mrcgen.field.XI0
XI1 = mrcgen.field.XI1
XI2 = mrcgen.field.XI2


def RHO_bc(dir):
    guu = TensorGUU()

    return (RHO(dir, 0, 0) - (XI0(dir) - XI0(0)) *
            (- guu[(0, 1)](dir, 0, 0) / guu[(0, 0)](dir, 0, 0) *
               (RHO(dir, 1, 0) - RHO(dir, -1, 0)) / (XI1(1) - XI1(-1))))


def T_bc(dir):
    guu = TensorGUU()
    GAM = mrcgen.field.GAM
    JAC = mrcgen.field.JAC

    fac = 1. + tau
    return (T(dir, 0, 0) - (XI0(dir) - XI0(0)) *
            (- guu[(0, 1)] / guu[(0, 0)] *
             0 * (T(0, 1, 0) - T(0, -1, 0)) / (XI1(1) - XI1(-1)) -
             (GAM(0, 0, 0) / (fac * JAC() * guu[(0, 0)]) * V[0] * V[0] +
              GAM(0, 0, 1) / (fac * JAC() * guu[(0, 0)]) * V[0] * V[1] +
              GAM(0, 1, 0) / (fac * JAC() * guu[(0, 0)]) * V[1] * V[0] +
              GAM(0, 1, 1) / (fac * JAC() * guu[(0, 0)]) * V[1] * V[1])
             )(dir, 0, 0))


print "########### WARNING T_bc"


def P_bc(dir):
    guu = TensorGUU()
    GAM = mrcgen.field.GAM
    Vcov = COV(V)

    V0 = P[0](dir, 0, 0) / RHO(dir, 0, 0)
    V0_g = -V0
    Vcov_g = [None] * 3

    Vcov_g[1] = (Vcov[1](dir, 0, 0) - (XI0(dir) - XI0(0)) *
                 (- guu[(0, 1)](dir, 0, 0) / guu[(0, 0)](dir, 0, 0) *
                  (Vcov[1](dir, 1, 0) - Vcov[1](dir, -1, 0)) /
                  (XI1(1) - XI1(-1)) +

                  guu[(0, 0)](dir, 0, 0) / guu[(0, 0)](dir, 0, 0) *
                  (GAM(0, 1, 0)(dir, 0, 0) * Vcov[0](dir, 0, 0) +
                   GAM(1, 1, 0)(dir, 0, 0) * Vcov[1](dir, 0, 0)) +

                  guu[(0, 1)](dir, 0, 0) / guu[(0, 0)](dir, 0, 0) *
                  (GAM(0, 1, 1)(dir, 0, 0) * Vcov[0](dir, 0, 0) +
                   GAM(1, 1, 1)(dir, 0, 0) * Vcov[1](dir, 0, 0))))

    Vcov_g[2] = (Vcov[2](dir, 0, 0) - (XI0(dir) - XI0(0)) *
                 (- guu[(0, 1)](dir, 0, 0) / guu[(0, 0)](dir, 0, 0) *
                  (Vcov[2](dir, 1, 0) - Vcov[2](dir, -1, 0)) /
                  (XI1(1) - XI1(-1))))

    Vcov_g[0] = (1. / guu[(0, 0)] *
                 (V0_g - guu[(0, 1)] * Vcov_g[1]))

    return [RHO * V0_g, RHO * CNV(Vcov_g)[1], RHO * CNV(Vcov_g)[2]]


def B_bc(dir):
    guu = TensorGUU()
    Bcov = COV(B)

    B0_g = (B[0](2 * dir, 0, 0) +
            (XI0(2 * dir) - XI0(0)) * (
                (B[1](dir, 1, 0) - B[1](dir, -1, 0)) / (XI1(1) - XI1(-1)) +
                (B[2](dir, 0, 1) - B[2](dir, 0, -1)) / (XI2(1) - XI2(-1))
    )
    )

    Bcov_g = [None] * 3
    Bcov_g[1] = (Bcov[1](dir, 0, 0) - (XI0(dir) - XI0(0)) *
                 (Bcov[0](dir, 1, 0) - Bcov[0](dir, -1, 0)) /
                 (XI1(1) - XI1(-1)))

    # + g^nk / g^nn ...
    Bcov_g[2] = (Bcov[2](dir, 0, 0) - (XI0(dir) - XI0(0)) *
                 (Bcov[0](dir, 0, 1) - Bcov[0](dir, 0, -1)) /
                 (XI2(1) - XI2(-1)))

    # + g^nk / g^nn ...
    # ??? What's this all about?
    Bcov_g[0] = (1. / guu[(0, 0)] *
                 (B0_g - guu[(0, 1)] * Bcov_g[1] - guu[(0, 2)] * Bcov_g[2]))

    return [B0_g, CNV(Bcov_g)[1], CNV(Bcov_g)[2]]


# BTYPE_SP is fundamentally broken (even the idea)
# but it's hacked into libmrc/src/mb.c for now.

RHO.bc = {"BTYPE_OUTER": [RHO_bc(1), RHO_bc(-1)],
          "BTYPE_SP": [RHO(1, 0, 0), RHO(-1, 0, 0)]}
T.bc = {"BTYPE_OUTER": [T_bc(1), T_bc(-1)],
        "BTYPE_SP": [T(1, 0, 0), T(-1, 0, 0)]}
P.bc = [{"BTYPE_OUTER": [P_bc(1)[0], P_bc(-1)[0]],
         "BTYPE_SP": [P[0](1, 0, 0), P[0](-1, 0, 0)]},
        {"BTYPE_OUTER": [P_bc(1)[1], P_bc(-1)[1]],
         "BTYPE_SP": [-P[1](1, 0, 0), -P[1](-1, 0, 0)]},
        {"BTYPE_OUTER": [P_bc(1)[2], P_bc(-1)[2]],
         "BTYPE_SP": [-P[2](1, 0, 0), -P[2](-1, 0, 0)]}]
B.bc = [{"BTYPE_OUTER": [B_bc(1)[0], B_bc(-1)[0]],
         "BTYPE_SP": [B[0](1, 0, 0), B[0](-1, 0, 0)]},
        {"BTYPE_OUTER": [B_bc(1)[1], B_bc(-1)[1]],
         "BTYPE_SP": [-B[1](1, 0, 0), -B[1](-1, 0, 0)]},
        {"BTYPE_OUTER": [B_bc(1)[2], B_bc(-1)[2]],
         "BTYPE_SP": [-B[2](1, 0, 0), -B[2](-1, 0, 0)]}]

j = VectorField("j", 0)


def j_bc(dir):
    guu = TensorGUU()

    j0_g = (j[0](2 * dir, 0, 0) +
            (XI0(2 * dir) - XI0(0)) * (
            (j[1](dir, 1, 0) - j[1](dir, -1, 0)) /
            (XI1(1) - XI1(-1)) +
            (j[2](dir, 0, 1) - j[2](dir, 0, -1)) /
            (XI2(1) - XI2(-1))
            )
            )

    jcnv1 = j0_g * guu[(0, 1)] / guu[(0, 0)]
    jcnv2 = j0_g * guu[(0, 2)] / guu[(0, 0)]

    j1_g = 2 * jcnv1 - j[1](dir, 0, 0)
    j2_g = 2 * jcnv2 - j[2](dir, 0, 0)

    return [j0_g, j1_g, j2_g]


j.bc = [{"BTYPE_OUTER": [j_bc(1)[0], j_bc(-1)[0]],
         "BTYPE_SP": [j[0](1, 0, 0), j[0](-1, 0, 0)]},
        {"BTYPE_OUTER": [j_bc(1)[1], j_bc(-1)[1]],
         "BTYPE_SP": [-j[1](1, 0, 0), -j[1](-1, 0, 0)]},
        {"BTYPE_OUTER": [j_bc(1)[2], j_bc(-1)[2]],
         "BTYPE_SP": [-j[2](1, 0, 0), -j[2](-1, 0, 0)]}]

if test_bc:
    B.bc = [{"BTYPE_OUTER": [5 * B[0](1, 0, 0), 5 * B[0](-1, 0, 0)]},
            {"BTYPE_OUTER": [B[1](1, 0, 0), B[1](-1, 0, 0)]},
            {"BTYPE_OUTER": [B[2](1, 0, 0), B[2](-1, 0, 0)]}, ]
    j.bc = [{"BTYPE_OUTER": [7 * j[0](1, 0, 0), 7 * j[0](-1, 0, 0)]},
            {"BTYPE_OUTER": [j[1](1, 0, 0), j[1](-1, 0, 0)]},
            {"BTYPE_OUTER": [j[2](1, 0, 0), j[2](-1, 0, 0)]}, ]

file = open("auto_mrc_3d_bc_%s.c" % (gname), "w")
file.write("  struct mrc_fld *x = l;\n")
file.write("  struct mrc_3d *ctx = mrc;\n")
file.write("  if (dir == 1) {\n")

btype = "BTYPE_OUTER"
file.write("    MRC_D5(x,RHO, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(RHO.bc[btype][0].at((0, 0, 0))))
file.write("    MRC_D5(x,T, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(T.bc[btype][0].at((0, 0, 0))))

for d in range(3):
    file.write("    MRC_D5(x,P%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(P.bc[d][btype][0].at((0, 0, 0))))

for d in range(3):
    file.write("    MRC_D5(x,B%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(B.bc[d][btype][0].at((0, 0, 0))))

file.write("  } else {\n")

file.write("    MRC_D5(x,RHO, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(RHO.bc[btype][1].at((0, 0, 0))))
file.write("    MRC_D5(x,T, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(T.bc[btype][1].at((0, 0, 0))))

for d in range(3):
    file.write("    MRC_D5(x,P%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(P.bc[d][btype][1].at((0, 0, 0))))

for d in range(3):
    file.write("    MRC_D5(x,B%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(B.bc[d][btype][1].at((0, 0, 0))))

file.write("  }\n")
file.close()

if sys.argv[1] == "MRC_CYLINDRICAL":
    VecSV = VectorSV_cyl
    TsrVV = TensorVV_cyl
    GrSV = GradSV_cyl
    DivMom = DivgAlt
    Curl_E = CURL
else:
    VecSV = VectorSV
    TsrVV = TensorVV
    GrSV = GradSV
    DivMom = Divg
    Curl_E = CURL

# density, temperature

rhs_RHO = 1. / J() * (-Divg(VecSV(RHO, V)) + D * DivGrad(RHO))
rhs_T = 1. / J() * (-Divg(VecSV(T, V)) -
                    (gamma - 2.) * T * Divg(V) + DT * DivGrad(T))

# momentum

T_rvv = .5 * (TsrVV(V, P) + TsrVV(P, V))
T_Lorentz = -TsrVV(B, B) + .5 * TensorBpress(guu(), B)
T_press = (1. + tau) * TensorTSS(J() * guu(), RHO, T)

rhs_P = -DivMom(T_rvv) - DivMom(0 * T_Lorentz) - \
    DivMom(T_press) - DivMom(-nu * GrSV(RHO, V))

# curl apparently works better than div tensor
rhs_P += CNV(CROSS(j, B))

# magnetic field

j_aux = CURL_cnv(B)
if test_bc:
    j_aux = [B[0](1, 0, 0) - B[0](-1, 0, 0), 0 * B[0], 0 * B[0]]
j_expr = [j_aux[0], j_aux[1], j_aux[2]]

# lapl_j = VectorField("lapl_j", 0)
# lapl_j_expr = [ DivGrad(j[0]), DivGrad(j[1]), DivGrad(j[2]) ]
gradpe = COV(DivMom(TensorTSS(J() * guu(), RHO, T)))

E = -CROSS(V, B) + d_i / RHO * (CROSS(j, B) - gradpe)
# + eta * COV(j)# - eta2 * COV(lapl_j)

if USE_HYPERRESISTIVITY:
    lapl_B = VectorField("lapl_B", 0)
    lapl_B_expr = [DivGradNum(B[0]), DivGradNum(B[1]), DivGradNum(B[2])]

    rhs_B = [-CURL(E)[0] - eta2 * DivGradNum(lapl_B[0]),
             -CURL(E)[1] - eta2 * DivGradNum(lapl_B[1]),
             -CURL(E)[2] - eta2 * DivGradNum(lapl_B[2])]
else:
    # CURL seems to give much better results at low res, but goes unstable
    # eventually..., so use CURL_COMP in cyl case for now.
    rhs_B = - Curl_E(E) - eta * CURL(COV(j))


# Trying to hack in a calculator for the electric field.
# Leaving out eta2 right now
Efld = CNV(-CROSS(V, B) + d_i / RHO * (CROSS(j, B) - gradpe) + eta * COV(j))
efld_expr = [Efld[0], Efld[1], Efld[2]]

# r.h.s.
rhs = [rhs_RHO,
       rhs_T,
       rhs_P[0], rhs_P[1], rhs_P[2],
       rhs_B[0], rhs_B[1], rhs_B[2]]

if test_bc:
    rhs = [B[0](1, 0, 0) - B[0](-1, 0, 0),
           (j[0](1, 0, 0) - j[0](-1, 0, 0))(0, -1, 0),
           0 * rhs_P[0], 0 * rhs_P[1], 0 * rhs_P[2],
           0 * rhs_B[0], 0 * rhs_B[1], 0 * rhs_B[2]]


# div B
divb = Divg(B)

divb_expr = [divb]

# curl
x = VectorField("x", 0)
curl_aux = CURL_cnv(x)
curl_expr = [curl_aux[0],
             curl_aux[1],
             curl_aux[2]]


# vorticity
om = VectorField("om", 0)

om_aux = CURL_cnv(P)
om_expr = [om_aux[0],
           om_aux[1],
           om_aux[2]]

# generate...
aux = []
aux.append(["j", j_expr, "bc_mrc_j"])

if USE_HYPERRESISTIVITY:
    aux.append(["lapl_B", lapl_B_expr, "bc_mrc_j"])
    # [ "lapl_j", lapl_j_expr, "bc_mrc_j" ] ],

sw = 2
bc = BC()
bc.l[0] = sw
bc.r[0] = sw
bc.btypes = ["BTYPE_OUTER"]  # , "BTYPE_SP" ]

gen = CodeGen("auto_mrc_3d_CalcRHS", rhs,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc",
              aux=aux,
              sub_src=True,
              sw=sw)  # FIXME, could be determined automatically

file = open("auto_mrc_3d_rhs_%s.c" % (gname), "w")
gen.function1(file, bc)
# gen.function(file)
# gen.timeloopFunction(file)
file.close()

file = open("auto_mrc_3d_jac_%s.c" % (gname), "w")
print "FIXME x"  # shouldn't have to pass x, and it just happens to work here
gen.createJacobian1("auto_mrc_3d_CreateJacobian", x, file, bc)
gen.calcJacobian1("auto_mrc_3d_CalcJacobian", x, file, bc)
file.close()

gen = CodeGen("auto_mrc_3d_CalcJ", j_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc")

file = open("auto_mrc_3d_j_%s.c" % (gname), "w")
gen.function(file)
file.close()

gen = CodeGen("auto_mrc_3d_CalcOm", om_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc")

file = open("auto_mrc_3d_om_%s.c" % (gname), "w")
gen.function(file)
file.close()

gen = CodeGen("auto_mrc_3d_CalcDivB", divb_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc")

file = open("auto_mrc_3d_divb_%s.c" % (gname), "w")
gen.function(file)
file.close()

gen = CodeGen("auto_mrc_3d_CalcCurl", curl_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc_extra")

file = open("auto_mrc_3d_curl_%s.c" % (gname), "w")
gen.function(file)
file.close()

# 4th order curl

x = VectorField("x", 0)
curl4_aux = CURL_fd4(COV(x))
curl4_expr = [curl4_aux[0],
              curl4_aux[1],
              curl4_aux[2]]

gen = CodeGen("auto_mrc_3d_CalcCurl_fd4", curl4_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc_extra", sw=2)

file = open("auto_mrc_3d_curl_fd4_%s.c" % (gname), "w")
gen.function(file)
file.close()

# I think we need this too
aux = []
aux.append(["j", j_expr, "bc_mrc_j"])

gen = CodeGen("auto_mrc_3d_CalcEfld", efld_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc",
              aux=aux)

file = open("auto_mrc_3d_calcefld_%s.c" % (gname), "w")
gen.function(file)
file.close()
