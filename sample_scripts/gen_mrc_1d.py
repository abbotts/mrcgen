#! /usr/bin/env python

import mrcgen.linear
from mrcgen.linear import *
from mrcgen.cgen import *

import sys

if len(sys.argv) < 2:
    raise ParseError

test_bc = False

if sys.argv[1] == "MRC_CARTESIAN":
    setTrafo(Trafo1dCartesian())
    gname = "linear_cart"
    # FIXME: Ugly...
    eqpath = "(mrc_to_subobj(ctx, struct mrc_3d_grid_linear_cart)->eq_ghost)"
# elif sys.argv[1] == "MRC_CYLINDRICAL":
#     setTrafo(Trafo2dCylindrical())
#     gname = "cylindrical"
# elif sys.argv[1] == "MRC_BUTTERFLY":
#     setTrafo(Trafo2d())
#     gname = "butterfly"
# else:
#     raise ParseError

USE_HYPERRESISTIVITY = False

D     = Parameter("D")
DT    = Parameter("DT")
nu    = Parameter("nu")
eta   = Parameter("eta")
eta2  = Parameter("eta2")
gamma = Parameter("gam")
tau   = Parameter("tau")
d_i   = Parameter("d_i")
#print "WARNING: d_i = 0"

# Equilibrium fields
RHO_0 = ScalarField(eqpath, 0)
T_0   = ScalarField(eqpath, 1)
P_0   = VectorField(eqpath, 2)
B_0   = VectorField(eqpath, 5)

V_0 = VectorSDiv(P_0, RHO_0)

# Perturbation Components

Re_RHO = ScalarField("x", 0)
Re_T   = ScalarField("x", 1)
Re_P   = VectorField("x", 2)
Re_B   = VectorField("x", 5)

Im_RHO = ScalarField("x", 8)
Im_T   = ScalarField("x", 9)
Im_P   = VectorField("x", 10)
Im_B   = VectorField("x", 13)

RHO = Re_RHO + I * Im_RHO
T = Re_T + I * Im_T
P = Re_P + I * Im_P
B = Re_B + I * Im_B
V = VectorSDiv(P, RHO_0)

#V = 1. / RHO * P

XI0 = mrcgen.linear.XI0
XI1 = mrcgen.linear.XI1
XI2 = mrcgen.linear.XI2

k = mrcgen.linear.k


j_0 = VectorField("j_0", 0)

########################################
# Boundary conditions on equilibrium fields
########################################

def RHO_0_bc(dir):
    guu = TensorGUU()

    #FIXME: Missing components for non-diagonal metric
    return RHO_0(dir,0,0)

def T_0_bc(dir):
    guu = TensorGUU()
    GAM = mrcgen.linear.GAM
    JAC = mrcgen.linear.JAC

    fac = 1. + tau
    
    # FIXME: Missing components for non-diagonal metric.

    return T_0(dir,0,0)


def P_0_bc(dir):
    guu = TensorGUU()
    GAM = mrcgen.linear.GAM
    Vcov = COV(V_0)
    
    V0 = P_0[0](dir,0,0) / RHO_0(dir,0,0)
    V0_g = -V0
    Vcov_g = [None] * 3

    # Still have a geometric contribution here, even though 1 and 2 derivative are 0
    Vcov_g[1] = (Vcov[1](dir,0,0) - (XI0(dir)-XI0(0)) *
                 (guu[(0,0)](dir,0,0) / guu[(0,0)](dir,0,0) *
                  (GAM(0,1,0)(dir,0,0) * Vcov[0](dir,0,0) +
                   GAM(1,1,0)(dir,0,0) * Vcov[1](dir,0,0))
                  + guu[(0,1)](dir,0,0) / guu[(0,0)](dir,0,0) *
                  (GAM(0,1,1)(dir,0,0) * Vcov[0](dir,0,0) +
                   GAM(1,1,1)(dir,0,0) * Vcov[1](dir,0,0))))
    Vcov_g[2] = (Vcov[2](dir,0,0))
    Vcov_g[0] = (1. / guu[(0,0)] *
                 (V0_g - guu[(0,1)] * Vcov_g[1]))

    return [ RHO_0 * V0_g, RHO_0 * CNV(Vcov_g)[1], RHO_0 * CNV(Vcov_g)[2] ]

def B_0_bc(dir):
    guu = TensorGUU()
    Bcov = COV(B_0)

    B0_g = B_0[0](2*dir,0,0) 
    #...

    Bcov_g = [None] * 3
    Bcov_g[1] = Bcov[1](dir,0,0)
    # + g^nk / g^nn ...
    Bcov_g[2] = Bcov[2](dir,0,0)
    #	  dir*d0*((Bcov(0, 0,jy,jz+1) - Bcov(0, 0,jy,jz-1))/(2.*d2))
    # + g^nk / g^nn ...
    # ??? What's this all about?
    Bcov_g[0] = (1./guu[(0,0)] *
                 (B0_g - guu[(0,1)] * Bcov_g[1] - guu[(0,2)] * Bcov_g[2]))

    return [ B0_g, CNV(Bcov_g)[1], CNV(Bcov_g)[2] ]

# BTYPE_SP is fundamentally broken (even the idea)
# but it's hacked into libmrc/src/mb.c for now.


RHO_0.bc =   { "BTYPE_OUTER" : [ RHO_0_bc(1)    , RHO_0_bc(-1)  ],
               "BTYPE_SP"    : [ RHO_0(1,0,0)   , RHO_0(-1,0,0) ] }
T_0.bc   =   { "BTYPE_OUTER" : [ T_0_bc(1)      , T_0_bc(-1)    ],
               "BTYPE_SP"    : [ T_0(1,0,0)     , T_0(-1,0,0)   ] }
P_0.bc   = [ { "BTYPE_OUTER" : [ P_0_bc(1)[0]   , P_0_bc(-1)[0]  ],
               "BTYPE_SP"    : [ P_0[0](1,0,0) , P_0[0](-1,0,0) ] },
             { "BTYPE_OUTER" : [ P_0_bc(1)[1]   , P_0_bc(-1)[1]  ],
               "BTYPE_SP"    : [ -P_0[1](1,0,0) , -P_0[1](-1,0,0) ] },
             { "BTYPE_OUTER" : [ P_0_bc(1)[2]   , P_0_bc(-1)[2]  ],
               "BTYPE_SP"    : [ -P_0[2](1,0,0) , -P_0[2](-1,0,0) ] } ]
B_0.bc   = [ { "BTYPE_OUTER" : [ B_0_bc(1)[0]   , B_0_bc(-1)[0]  ],
               "BTYPE_SP"    : [ B_0[0](1,0,0)  , B_0[0](-1,0,0) ] },
             { "BTYPE_OUTER" : [ B_0_bc(1)[1]   , B_0_bc(-1)[1]  ],
               "BTYPE_SP"    : [ -B_0[1](1,0,0) , -B_0[1](-1,0,0) ] },
             { "BTYPE_OUTER" : [ B_0_bc(1)[2]   , B_0_bc(-1)[2]  ],
               "BTYPE_SP"    : [ -B_0[2](1,0,0) , -B_0[2](-1,0,0) ] } ]


def j_0_bc(dir):
    guu = TensorGUU()

    j0_g = j_0[0](2*dir,0,0)


    jcnv1 = j0_g * guu[(0,1)] / guu[(0,0)]
    jcnv2 = j0_g * guu[(0,2)] / guu[(0,0)]

    j1_g = 2*jcnv1 - j_0[1](dir,0,0)
    j2_g = 2*jcnv2 - j_0[2](dir,0,0)

    return [ j0_g, j1_g, j2_g ]

j_0.bc   = [ { "BTYPE_OUTER" : [ j_0_bc(1)[0] , j_0_bc(-1)[0]  ],
               "BTYPE_SP"    : [ j_0[0](1,0,0), j_0[0](-1,0,0) ] },
             { "BTYPE_OUTER" : [ j_0_bc(1)[1] , j_0_bc(-1)[1]  ],
               "BTYPE_SP"    : [ -j_0[1](1,0,0), -j_0[1](-1,0,0) ] },
             { "BTYPE_OUTER" : [ j_0_bc(1)[2] , j_0_bc(-1)[2]  ],
               "BTYPE_SP"    : [ -j_0[2](1,0,0), -j_0[2](-1,0,0) ] } ]


file = open("auto_mrc_3d_bc_%s_eq.c" %(gname), "w")
file.write("  struct mrc_fld *x = l;\n")
file.write("  struct mrc_3d *ctx = mrc;\n")
file.write("  if (dir == 1) {\n")

btype = "BTYPE_OUTER"
file.write("    MRC_D5(x,RHO, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(RHO_0.bc[btype][0].at((0,0,0))))
file.write("    MRC_D5(x,T, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(T_0.bc[btype][0].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,P%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(P_0.bc[d][btype][0].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,B%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(B_0.bc[d][btype][0].at((0,0,0))))

file.write("  } else {\n")

file.write("    MRC_D5(x,RHO, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(RHO_0.bc[btype][1].at((0,0,0))))
file.write("    MRC_D5(x,T, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(T_0.bc[btype][1].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,P%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(P_0.bc[d][btype][1].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,B%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(B_0.bc[d][btype][1].at((0,0,0))))

file.write("  }\n")
file.close()


########################################
# Boundary conditions on perturbed fields
########################################

def RHO_bc(dir):
    guu = TensorGUU()

    #FIXME: Missing components for non-diagonal metric
    return RHO(dir,0,0)

def T_bc(dir):
    guu = TensorGUU()
    GAM = mrcgen.linear.GAM
    JAC = mrcgen.linear.JAC

    fac = 1. + tau
    
    # FIXME: Missing components for non-diagonal metric.

    return T(dir,0,0)


def P_bc(dir):
    guu = TensorGUU()
    GAM = mrcgen.linear.GAM
    Vcov = COV(V)
    
    V0 = P[0](dir,0,0) / RHO_0(dir,0,0)
    V0_g = -V0
    Vcov_g = [None] * 3

    Vcov_g[1] = (Vcov[1](dir,0,0) - (XI0(dir)-XI0(0)) *
                 (- guu[(0,1)](dir,0,0) / guu[(0,0)](dir,0,0) *
                    I*k[1]*Vcov[1](dir,0,0)
                  + guu[(0,0)](dir,0,0) / guu[(0,0)](dir,0,0) *
                    (GAM(0,1,0)(dir,0,0) * Vcov[0](dir,0,0) +
                     GAM(1,1,0)(dir,0,0) * Vcov[1](dir,0,0))
                  + guu[(0,1)](dir,0,0) / guu[(0,0)](dir,0,0) *
                    (GAM(0,1,1)(dir,0,0) * Vcov[0](dir,0,0) +
                     GAM(1,1,1)(dir,0,0) * Vcov[1](dir,0,0))))
    Vcov_g[2] = (Vcov[2](dir,0,0) - (XI0(dir)-XI0(0)) *
                 (- guu[(0,1)](dir,0,0) / guu[(0,0)](dir,0,0) *
                    I*k[1]*Vcov[2](dir,0,0)))
    Vcov_g[0] = (1. / guu[(0,0)] *
                 (V0_g - guu[(0,1)] * Vcov_g[1]))

    return [ RHO_0 * V0_g, RHO_0 * CNV(Vcov_g)[1], RHO_0 * CNV(Vcov_g)[2] ]

def B_bc(dir):
    guu = TensorGUU()
    Bcov = COV(B)

    B0_g = (B[0](2*dir,0,0) +
            (XI0(2*dir)-XI0(0)) * (
                I*k[1]*B[1](dir,0,0)
                + I*k[2]*B[2](dir,0,0)
                )
        )
    #...

    Bcov_g = [None] * 3
    Bcov_g[1] = (Bcov[1](dir,0,0) - (XI0(dir)-XI0(0))
                 *I*k[1]*Bcov[0](dir,0,0))
    # + g^nk / g^nn ...
    Bcov_g[2] = (Bcov[2](dir,0,0) - (XI0(dir)-XI0(0))
                 *I*k[2]*Bcov[0](dir,0,0))
    #	  dir*d0*((Bcov(0, 0,jy,jz+1) - Bcov(0, 0,jy,jz-1))/(2.*d2))
    # + g^nk / g^nn ...
    # ??? What's this all about?
    Bcov_g[0] = (1./guu[(0,0)] *
                 (B0_g - guu[(0,1)] * Bcov_g[1] - guu[(0,2)] * Bcov_g[2]))

    return [ B0_g, CNV(Bcov_g)[1], CNV(Bcov_g)[2] ]

# BTYPE_SP is fundamentally broken (even the idea)
# but it's hacked into libmrc/src/mb.c for now.


Re_RHO.bc =   { "BTYPE_OUTER" : [ Real(RHO_bc(1)) , Real(RHO_bc(-1))  ],
                "BTYPE_SP"    : [ Real(RHO(1,0,0)), Real(RHO(-1,0,0)) ] }
Im_RHO.bc =   { "BTYPE_OUTER" : [ Imag(RHO_bc(1)) , Imag(RHO_bc(-1))  ],
                "BTYPE_SP"    : [ Imag(RHO(1,0,0)), Imag(RHO(-1,0,0)) ] }
Re_T.bc   =   { "BTYPE_OUTER" : [ Real(T_bc(1))   , Real(T_bc(-1))    ],
                "BTYPE_SP"    : [ Real(T(1,0,0))  , Real(T(-1,0,0))   ] }
Im_T.bc   =   { "BTYPE_OUTER" : [ Imag(T_bc(1))   , Imag(T_bc(-1))    ],
                "BTYPE_SP"    : [ Imag(T(1,0,0))  , Imag(T(-1,0,0))   ] }
Re_P.bc   = [ { "BTYPE_OUTER" : [ Real(P_bc(1)[0]) , Real(P_bc(-1)[0])  ],
                "BTYPE_SP"    : [ Real(P[0](1,0,0)), Real(P[0](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Real(P_bc(1)[1]) , Real(P_bc(-1)[1])  ],
                "BTYPE_SP"    : [ Real(-P[1](1,0,0)), Real(-P[1](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Real(P_bc(1)[2]) , Real(P_bc(-1)[2])  ],
                "BTYPE_SP"    : [ Real(-P[2](1,0,0)), Real(-P[2](-1,0,0)) ] } ]
Im_P.bc   = [ { "BTYPE_OUTER" : [ Imag(P_bc(1)[0]) , Imag(P_bc(-1)[0])  ],
                "BTYPE_SP"    : [ Imag(P[0](1,0,0)), Imag(P[0](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Imag(P_bc(1)[1]) , Imag(P_bc(-1)[1])  ],
                "BTYPE_SP"    : [ Imag(-P[1](1,0,0)), Imag(-P[1](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Imag(P_bc(1)[2]) , Imag(P_bc(-1)[2])  ],
                "BTYPE_SP"    : [ Imag(-P[2](1,0,0)), Imag(-P[2](-1,0,0)) ] } ]
Re_B.bc   = [ { "BTYPE_OUTER" : [ Real(B_bc(1)[0]) , Real(B_bc(-1)[0])  ],
                "BTYPE_SP"    : [ Real(B[0](1,0,0)), Real(B[0](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Real(B_bc(1)[1]) , Real(B_bc(-1)[1])  ],
                "BTYPE_SP"    : [ Real(-B[1](1,0,0)), Real(-B[1](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Real(B_bc(1)[2]) , Real(B_bc(-1)[2])  ],
                "BTYPE_SP"    : [ Real(-B[2](1,0,0)), Real(-B[2](-1,0,0)) ] } ]
Im_B.bc   = [ { "BTYPE_OUTER" : [ Imag(B_bc(1)[0]) , Imag(B_bc(-1)[0])  ],
                "BTYPE_SP"    : [ Imag(B[0](1,0,0)), Imag(B[0](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Imag(B_bc(1)[1]) , Imag(B_bc(-1)[1])  ],
                "BTYPE_SP"    : [ Imag(-B[1](1,0,0)), Imag(-B[1](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Imag(B_bc(1)[2]) , Imag(B_bc(-1)[2])  ],
                "BTYPE_SP"    : [ Imag(-B[2](1,0,0)), Imag(-B[2](-1,0,0)) ] } ]



Re_j = VectorField("j", 0)
Im_j = VectorField("j", 3)


j = Re_j + I * Im_j

def j_bc(dir):
    guu = TensorGUU()

    j0_g = (j[0](2*dir,0,0) +
            (XI0(2*dir)-XI0(0)) * (
                I*k[1]*j[1](dir,0,0)
                + I*k[2]*j[2](dir,0,0)
            )
        )


    jcnv1 = j0_g * guu[(0,1)] / guu[(0,0)]
    jcnv2 = j0_g * guu[(0,2)] / guu[(0,0)]

    j1_g = 2*jcnv1 - j[1](dir,0,0)
    j2_g = 2*jcnv2 - j[2](dir,0,0)

    return [ j0_g, j1_g, j2_g ]

Re_j.bc   = [ { "BTYPE_OUTER" : [ Real(j_bc(1)[0]) , Real(j_bc(-1)[0])  ],
                "BTYPE_SP"    : [ Real(j[0](1,0,0)), Real(j[0](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Real(j_bc(1)[1]) , Real(j_bc(-1)[1])  ],
                "BTYPE_SP"    : [ Real(-j[1](1,0,0)), Real(-j[1](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Real(j_bc(1)[2]) , Real(j_bc(-1)[2])  ],
                "BTYPE_SP"    : [ Real(-j[2](1,0,0)), Real(-j[2](-1,0,0)) ] } ]

Im_j.bc   = [ { "BTYPE_OUTER" : [ Imag(j_bc(1)[0]) , Imag(j_bc(-1)[0])  ],
                "BTYPE_SP"    : [ Imag(j[0](1,0,0)), Imag(j[0](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Imag(j_bc(1)[1]) , Imag(j_bc(-1)[1])  ],
                "BTYPE_SP"    : [ Imag(-j[1](1,0,0)), Imag(-j[1](-1,0,0)) ] },
              { "BTYPE_OUTER" : [ Imag(j_bc(1)[2]) , Imag(j_bc(-1)[2])  ],
                "BTYPE_SP"    : [ Imag(-j[2](1,0,0)), Imag(-j[2](-1,0,0)) ] } ]


file = open("auto_mrc_3d_bc_%s.c" %(gname), "w")
file.write("  struct mrc_fld *x = l;\n")
file.write("  struct mrc_3d *ctx = mrc;\n")
file.write("  if (dir == 1) {\n")

btype = "BTYPE_OUTER"
file.write("    MRC_D5(x,RHO, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(Re_RHO.bc[btype][0].at((0,0,0))))
file.write("    MRC_D5(x,T, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(Re_T.bc[btype][0].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,P%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(Re_P.bc[d][btype][0].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,B%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(Re_B.bc[d][btype][0].at((0,0,0))))
file.write("    MRC_D5(x,RHO+8, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(Im_RHO.bc[btype][0].at((0,0,0))))
file.write("    MRC_D5(x,T+8, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(Im_T.bc[btype][0].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,P%d+8, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(Im_P.bc[d][btype][0].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,B%d+8, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(Im_B.bc[d][btype][0].at((0,0,0))))

file.write("  } else {\n")

file.write("    MRC_D5(x,RHO, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(Re_RHO.bc[btype][1].at((0,0,0))))
file.write("    MRC_D5(x,T, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(Re_T.bc[btype][1].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,P%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(Re_P.bc[d][btype][1].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,B%d, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(Re_B.bc[d][btype][1].at((0,0,0))))
file.write("    MRC_D5(x,RHO+8, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(Im_RHO.bc[btype][1].at((0,0,0))))
file.write("    MRC_D5(x,T+8, jx,jy,jz,patch) =\n")
file.write("      %s;\n" % gen_C(Im_T.bc[btype][1].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,P%d+8, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(Im_P.bc[d][btype][1].at((0,0,0))))
for d in range(3):
    file.write("    MRC_D5(x,B%d+8, jx,jy,jz,patch) =\n" % d)
    file.write("      %s;\n" % gen_C(Im_B.bc[d][btype][1].at((0,0,0))))

file.write("  }\n")
file.close()

########################################
# RHS Equations
########################################

if sys.argv[1] == "MRC_CYLINDRICAL":
    VecSV = VectorSV_cyl
    TsrVV = TensorVV_cyl
    GrSV = GradSV_cyl
    DivMom = DivgAlt
    Curl_E = CURL #_COMP
else:
    VecSV = VectorSV
    TsrVV = TensorVV
    GrSV = GradSV
    GrSV_EQ = GradSV_EQ
    DivMom = Divg
    Curl_E = CURL

# density, temperature

rhs_RHO = 1./J() * (-Divg(VecSV(RHO, V_0)) - Divg(VecSV(RHO_0, V)) 
                    + D*DivGrad(RHO))

rhs_T   = 1./J() * (-Divg(VecSV(T  , V_0)) - Divg(VecSV(T_0  , V))
                    - (gamma - 2.)*T_0*Divg(V) - (gamma - 2.)*T*EQDivg(V_0)
                    + DT*DivGrad(T))

# momentum

T_rvv     = .5*(TsrVV(V_0, P) + TsrVV(P_0, V) + TsrVV(V, P_0) + TsrVV(P, V_0))
T_Lorentz = - TsrVV(B_0, B) - TsrVV(B, B_0) + .5*TensorBpress(guu(), B) # Hmm.. this may not be right...
T_press   = (1.+tau) * TensorTSS(J()*guu(), RHO_0, T) + (1.+tau) * TensorTSS(J()*guu(), RHO, T_0) 

rhs_P = - DivMom(T_rvv) - DivMom(0*T_Lorentz)  - DivMom(T_press)  - DivMom(- nu*GrSV(RHO_0,V)) - DivMom(- nu*GrSV_EQ(RHO,V_0))


# curl apparently works better than div tensor 
rhs_P += CNV(CROSS(j_0, B)) + CNV(CROSS(j, B_0))

# magnetic field

j_aux  = CURL_cnv(B)

j_expr = [ Real(j_aux[0]), 
           Real(j_aux[1]), 
           Real(j_aux[2]),
           Imag(j_aux[0]), 
           Imag(j_aux[1]), 
           Imag(j_aux[2]) ]

j_0_aux = CURL_EQ(COV(B_0))
j_0_expr = [ j_0_aux[0], j_0_aux[1], j_0_aux[2] ]

#lapl_j = VectorField("lapl_j", 0)
#lapl_j_expr = [ DivGrad(j[0]), DivGrad(j[1]), DivGrad(j[2]) ]
gradpe = COV(DivMom(TensorTSS(J()*guu(), RHO_0, T))) + COV(DivMom(TensorTSS(J()*guu(), RHO, T_0)))
EQgradpe = COV(DivMom(TensorTSS(J()*guu(), RHO_0, T_0))) + COV(DivMom(TensorTSS(J()*guu(), RHO, T_0)))

E = ( - CROSS(V_0, B) - CROSS(V, B_0)
      + d_i / RHO_0 * (CROSS(j_0, B) + CROSS(j, B_0) - gradpe 
                       - RHO / RHO_0 *(CROSS(j_0, B_0) - COV(EQVecDivg(TensorTSS(J()*guu(), RHO_0, T_0)))))) 

# + eta * COV(j)# - eta2 * COV(lapl_j)

# if USE_HYPERRESISTIVITY:
#     lapl_B = VectorField("lapl_B", 0)
#     lapl_B_expr = [ DivGradNum(B[0]), DivGradNum(B[1]), DivGradNum(B[2]) ]
    
#     rhs_B = [ - CURL(E)[0] - eta2 * DivGradNum(lapl_B[0]),
#               - CURL(E)[1] - eta2 * DivGradNum(lapl_B[1]),
#               - CURL(E)[2] - eta2 * DivGradNum(lapl_B[2]) ]
# else:
# CURL seems to give much better results at low res, but goes unstable
# eventually..., so use CURL_COMP in cyl case for now.
rhs_B = - Curl_E(E) - eta * CURL(COV(j))
    

# Trying to hack in a calculator for the electric field. Leaving out eta2 right now
Efld = CNV(E + eta * COV(j))
efld_expr = [Real(Efld[0]), 
             Real(Efld[1]), 
             Real(Efld[2]),
             Imag(Efld[0]), 
             Imag(Efld[1]), 
             Imag(Efld[2])]
# r.h.s.

rhs = [ Real(rhs_RHO),
        Real(rhs_T),
        Real(rhs_P[0]), Real(rhs_P[1]), Real(rhs_P[2]),
        Real(rhs_B[0]), Real(rhs_B[1]), Real(rhs_B[2]),
        Imag(rhs_RHO),
        Imag(rhs_T),
        Imag(rhs_P[0]), Imag(rhs_P[1]), Imag(rhs_P[2]),
        Imag(rhs_B[0]), Imag(rhs_B[1]), Imag(rhs_B[2]) ]




aux = []
aux.append([ "j"       , j_expr       , "bc_mrc_j" ])
aux.append([ "j_0"     , j_0_expr     , "bc_mrc_j" ])

if USE_HYPERRESISTIVITY:
    aux.append([ "lapl_B", lapl_B_expr, "bc_mrc_j" ])
    #[ "lapl_j", lapl_j_expr, "bc_mrc_j" ] ],

sw = 2
bc = BC()
bc.l[0] = sw
bc.r[0] = sw
bc.btypes = [ "BTYPE_OUTER" ]#, "BTYPE_SP" ]

gen = CodeGen("auto_mrc_3d_CalcRHS", rhs, 
              ctx_type="struct mrc_3d *",
              bc="bc_mrc",
              aux=aux,
              sub_src=True,
              sw=sw) # FIXME, could be determined automatically

file = open("auto_mrc_3d_rhs_%s.c" %(gname), "w")
gen.function1(file, bc)
#gen.function(file)
#gen.timeloopFunction(file)
file.close()

x = VectorField("x", 0)
file = open("auto_mrc_3d_jac_%s.c" %(gname), "w")
print "FIXME x" # shouldn't have to pass x, and it just happens to work here
gen.createJacobian1("auto_mrc_3d_CreateJacobian", x, file, bc)
gen.calcJacobian1("auto_mrc_3d_CalcJacobian", x, file, bc)
file.close()


# div B

divb = Divg(B)

divb_expr = [ Real(divb), Imag(divb) ] 

# curl

vecA = VectorField("x", 0) + I * VectorField("x", 3)
curl_aux  = CURL_cnv(vecA)
curl_expr = [ Real(curl_aux[0]),
              Real(curl_aux[1]),
              Real(curl_aux[2]),
              Imag(curl_aux[0]),
              Imag(curl_aux[1]),
              Imag(curl_aux[2]) ]



# vorticity

om = VectorField("om", 0)

om_aux  = CURL_cnv(P)
om_expr = [ Real(om_aux[0]),
            Real(om_aux[1]),
            Real(om_aux[2]),
            Imag(om_aux[0]),
            Imag(om_aux[1]),
            Imag(om_aux[2]) ]
# generate...


gen = CodeGen("auto_mrc_3d_CalcJ", j_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc")

file = open("auto_mrc_3d_j_%s.c" %(gname), "w")
gen.function(file)
file.close()

gen = CodeGen("auto_mrc_3d_CalcOm", om_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc")

file = open("auto_mrc_3d_om_%s.c" %(gname), "w")
gen.function(file)
file.close()

gen = CodeGen("auto_mrc_3d_CalcDivB", divb_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc")

file = open("auto_mrc_3d_divb_%s.c" %(gname), "w")
gen.function(file)
file.close()

gen = CodeGen("auto_mrc_3d_CalcCurl", curl_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc_extra")

file = open("auto_mrc_3d_curl_%s.c" %(gname), "w")
gen.function(file)
file.close()

# 4th order curl

x = VectorField("x", 0) + I * VectorField("x", 3)
curl4_aux  = CURL_fd4(COV(x))
curl4_expr = [ Real(curl4_aux[0]),
               Real(curl4_aux[1]),
               Real(curl4_aux[2]),
               Imag(curl4_aux[0]),
               Imag(curl4_aux[1]),
               Imag(curl4_aux[2])
           ]

gen = CodeGen("auto_mrc_3d_CalcCurl_fd4", curl4_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc_extra",sw=2)

file = open("auto_mrc_3d_curl_fd4_%s.c" %(gname), "w")
gen.function(file)
file.close()

# I think we need this too
aux = []
aux.append([ "j"       , j_expr       , "bc_mrc_j" ])
aux.append([ "j_0"     , j_0_expr     , "bc_mrc_j" ])


gen = CodeGen("auto_mrc_3d_CalcEfld", efld_expr,
              ctx_type="struct mrc_3d *",
              bc="bc_mrc",
              aux=aux)

file = open("auto_mrc_3d_calcefld_%s.c" %(gname), "w")
gen.function(file)
file.close()
