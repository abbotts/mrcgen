from sympy import *
from sympy.utilities.iterables import postorder_traversal
from field import F3

# ======================================================================
# generate C-style expressions

from sympy.printing.printer import Printer
from sympy.printing.ccode import CCodePrinter


class CPrinter(CCodePrinter):
    def _print_Pow(self, expr):
        if expr.exp == 2:
            return "((%s)*(%s))" % (self._print(expr.base), self._print(expr.base))
        if expr.exp == -1:
            return "(1./(%s))" % (self._print(expr.base))
        return "pow(%s,%s)" % (self._print(expr.base), self._print(expr.exp))


def gen_C(expr):
    return CPrinter().doprint(expr)

######################################################################


class BC_Face:
    def __init__(self, fld, expr):
        self.fld = fld
        self.expr = expr

    def __repr__(self):
        return "fld = " + str(self.fld) + ", expr = " + str(self.expr)


class BC:
    FACE_LEFT = 0
    FACE_RIGHT = 1
    NR_FACES = 2

    def __init__(self):
        self.bcs = [None] * BC.NR_FACES
        self.l = [0, 0, 0]
        self.r = [0, 0, 0]

    def setBC(self, face, fld, expr):
        self.bcs[face] = BC_Face(fld, expr)

    def __repr__(self):
        return str(self.bcs) + "\n" + "l = " + str(self.l) + ", r = " + str(self.r)


######################################################################

def subst_left(expr, idx, btype):
    s = dict()
    for e in postorder_traversal(expr):
        if type(e) == F3:
            if idx + e.idx[0] == -1:
                s[e] = e.fld.bc[btype][0].at((-1 - idx, e.idx[1], e.idx[2]))

    if len(s):
        expr = expr.subs(s)

    return expr


def subst_right(expr, idx, btype):
    s = dict()
    for e in postorder_traversal(expr):
        if type(e) == F3:
            if e.idx[0] - idx == 1:
                s[e] = e.fld.bc[btype][1].at((1 + idx, e.idx[1], e.idx[2]))

    if len(s):
        expr = expr.subs(s)

    return expr


def subst_aux(expr, aux):
    if not aux:
        return expr

    s = dict()
    for e in postorder_traversal(expr):
        # FIXME, this should really be an isinstance off a shared ancestor type
        # or better yet some sort of duck typing thing
        if type(e) == F3:
            for a in aux:
                if e.fld.str == a[0]:
                    s[e] = a[1][e.fld.m].at(e.idx)

    if len(s):
        expr = expr.subs(s)

    return expr


######################################################################

class CodeGen:
    # ------------------------------------------------------------------
    def __init__(self, name, func, bc=None, ctx_type="struct my *",
                 aux=None, sub_src=False, sw=1):
        self._name, self._func = name, func
        self._bc = bc
        self._sw = sw
        self._ctx_type = ctx_type
        self._bs = len(func)
        self._aux = aux
        self._sub_src = sub_src

    # ------------------------------------------------------------------
    def getBackSubstExprs(self, dir, idx, btype):
        exprs = []

        for f in self._func:
            exprs.append(f.at((0, 0, 0)))

        while 1:
            exprs2 = []
            for expr in exprs:
                if dir == -1:
                    expr = subst_left(expr, idx, btype)
                elif dir == +1:
                    expr = subst_right(expr, idx, btype)

                exprs2.append(expr)

            exprs = []
            done = True
            for expr2 in exprs2:
                expr = subst_aux(expr2, self._aux)
                if not expr is expr2:
                    done = False
                exprs.append(expr)

            if done:
                return exprs

    # ------------------------------------------------------------------
    def genLoopBegin(self, file, dir, idx, btype):

        if dir == -1:
            file.write("""
  for (int jz = 0; jz < info.ldims[2]; jz++) {
    for (int jy = 0; jy < info.ldims[1]; jy++) {
      int jx = %d; {
""" % idx)
        elif dir == +1:
            file.write("""
  for (int jz = 0; jz < info.ldims[2]; jz++) {
    for (int jy = 0; jy < info.ldims[1]; jy++) {
      int jx = info.ldims[0] - 1 - %d; {
""" % idx)
        else:
            file.write("""
  int sw_l = %d, sw_r = %d;
  if (info.p_pface[FACE_LEFT].pf_btype <= BTYPE_PATCH ||
      info.p_pface[FACE_LEFT].pf_btype == BTYPE_SP ) {
    sw_l = 0;
  } 
  if (info.p_pface[FACE_RIGHT].pf_btype <= BTYPE_PATCH ||
      info.p_pface[FACE_RIGHT].pf_btype == BTYPE_SP ) {
    sw_r = 0;
  }
  for (int jz = 0; jz < info.ldims[2]; jz++) {
    for (int jy = 0; jy < info.ldims[1]; jy++) {
      for (int jx = sw_l; jx < info.ldims[0] - sw_r; jx++) {
""" % idx)

    # ------------------------------------------------------------------
    def genLoopEnd(self, file, dir, idx):
        file.write("""
      }
    }
  }""")

    # ------------------------------------------------------------------
    def localFunction(self, file, dir, idx, btype=None):
        """generates code for the function evaluation"""

        file.write("static int\n")
        file.write('__attribute__((optimize("no-var-tracking-assignments")))\n')
        if dir == -1:
            suffix = "local_l_%d_%s" % (idx, btype)
        elif dir == +1:
            suffix = "local_r_%d_%s" % (idx, btype)
        else:
            suffix = "local"

        file.write("%s_%s(%s ctx, int patch, struct mrc_fld *x, struct mrc_fld *f)\n" %
                   (self._name, suffix, self._ctx_type))
        file.write("{\n")
        file.write("struct mrc_crds *crds = mrc_domain_get_crds(ctx->mb);\n")
        file.write("struct mrc_trafo *trafo = ctx->trafo;\n")
        file.write("struct mrc_patch_info info;\n")
        file.write("mrc_domain_get_local_patch_info(ctx->mb, patch, &info);\n")
        file.write("  pfb;\n")

        self.genLoopBegin(file, dir, idx, btype)

        exprs = self.getBackSubstExprs(dir, idx, btype)

        if 0:
            sub, expr_list = cse(exprs,
                                 cse_main.numbered_symbols('__cse_'))
            for sub_expr in sub:
                file.write("double %s = %s;\n" %
                           (sub_expr[0], gen_C(sub_expr[1])))

            file.write("\n")

            for i in xrange(len(expr_list)):
                file.write('MRC_D5(f,%d, jx,jy,jz,patch) = %s;\n\n' %
                           (i, gen_C(expr_list[i])))

        else:
            for i in xrange(len(exprs)):
                file.write('MRC_D5(f,%d, jx,jy,jz,patch) = %s;\n\n' %
                           (i, gen_C(exprs[i])))

        self.genLoopEnd(file, dir, idx)

        file.write("  \n")
        file.write("  pfr;\n")
        file.write("}\n\n")

    # ------------------------------------------------------------------
    def function1(self, file, bc):
        """generates code for the function evaluation"""

        self.localFunction(file, 0, (bc.l[0], bc.r[0]))
        for btype in bc.btypes:
            for i in range(0, bc.l[0]):
                self.localFunction(file, -1, i, btype)
            for i in range(0, bc.r[0]):
                self.localFunction(file, +1, i, btype)

        if self._bc:
            sbc = self._bc
        else:
            sbc = "NULL"

        file.write("""
static void
%s(void *_ctx, struct mrc_obj *_f, float time, struct mrc_obj *_xg)
{
  int ierr;
  %s ctx = (%s) _ctx;
  struct mrc_fld *xg = (struct mrc_fld *) _xg;
  struct mrc_fld *f = (struct mrc_fld *) _f;
  struct mrc_domain *mb = xg->_domain;
  struct mrc_trafo *trafo = ctx->trafo; // FIXME Not general
  struct mrc_crds *crds = mrc_domain_get_crds(mb);
  assert(mrc_fld_nr_comps(f) == %d);
""" % (self._name, self._ctx_type, self._ctx_type, self._bs))

        file.write("""
  struct mrc_fld *x = mrc_fld_create(mrc_domain_comm(mb));
  MB_GetFld(xg->_domain, mrc_fld_nr_comps(xg), %d, x);
  M3d_FillGhostCells(ctx->bnd, xg, x, \"%s\");
  if (ctx->debug_rhs) {
    static int cnt = 0;
    char fn[255];
    sprintf(fn, "A_%%06d_x", cnt++);
    mrc_3d_Output(ctx, x, fn);
  }

""" % (self._sw, sbc))

        file.write("  int nr_local_patches;\n")
        file.write("  mrc_domain_get_patches(mb, &nr_local_patches);\n")
        file.write("  for(int patch=0; patch < nr_local_patches; patch++) {\n")
        file.write("    struct mrc_patch_info info;\n")
        file.write("    mrc_domain_get_local_patch_info(mb, patch, &info);\n")

        for btype in bc.btypes:
            file.write(
                "    if (info.p_pface[FACE_LEFT].pf_btype == %s) {\n" % btype)
            for i in range(0, bc.l[0]):
                file.write("      ierr = %s_local_l_%d_%s(ctx, patch, x, f); CE;\n"
                           % (self._name, i, btype))
            file.write("    }\n")

        file.write("    ierr = %s_local(ctx, patch, x, f); CE;\n"
                   % (self._name))

        for btype in bc.btypes:
            file.write(
                "    if (info.p_pface[FACE_RIGHT].pf_btype == %s) {\n" % btype)
            for i in reversed(range(0, bc.r[0])):
                file.write("      ierr = %s_local_r_%d_%s(ctx, patch, x, f); CE;\n"
                           % (self._name, i, btype))
            file.write("    }\n")

        file.write("  }\n")
        file.write("""   if (ctx->debug_rhs) { \n
    static int cnt = 0; \n
    char fn[255]; \n
    sprintf(fn, "A_%06d", cnt++);\n
    mrc_3d_Output(ctx, f, fn); \n
        }\n""")

        if self._sub_src:
            file.write("""
    if (ctx->src){
     if (mrc_fld_is_setup(ctx->src)) {
      void (*fld_axpy)(struct mrc_fld *, float, struct mrc_fld *);
      fld_axpy = (void (*)(struct mrc_fld *, float, struct mrc_fld *)) mrc_fld_get_method(f, "axpy");
      fld_axpy(f, -1., ctx->src);
    }
  }
""")

        file.write("""
  mrc_fld_destroy(x);
}

""")

    # ------------------------------------------------------------------
    def snesFunction(self, file):
        """generates code for the function evaluation wrapper"""

        file.write("""
static int
__attribute__((optimize("no-var-tracking-assignments")))
snes_%s(SNES snes, Vec _x, Vec _f, void *_ctx)
{
  int ierr;
  %s ctx = (%s) _ctx;

  pfb;
  struct mrc_fld *x = mrc_fld_create(mrc_domain_comm(ctx->mb));
  struct mrc_fld *f = mrc_fld_create(mrc_domain_comm(ctx->mb));
  ierr = MB_GetFldWithVec(ctx->mb, 0, _x, x); CE;
  ierr = MB_GetFldWithVec(ctx->mb, 0, _f, f); CE;
  ierr = %s(ctx, x, f); CE;
  mrc_fld_destroy(x);
  mrc_fld_destroy(f);
  pfr;
}
""" % (self._name, self._ctx_type, self._ctx_type, self._name))

    # ------------------------------------------------------------------
    def timeloopFunction(self, file):
        """generates code for the timeloop wrapper"""

        file.write("""
static int
timeloop_%s(Vec _r, Vec _x, struct timeloop_ctx *_ctx)
{
  int ierr;
  %s ctx = (%s) _ctx->ctx;

  // FIXME: Assumes ctx object has a time attribute
  ctx->time = _ctx->time;
  pfb;
  // FIXME: Code
  struct mrc_fld *x = mrc_fld_create(mrc_domain_comm(ctx->mb));
  struct mrc_fld *r = mrc_fld_create(mrc_domain_comm(ctx->mb));
  ierr = MB_GetFldWithVec(ctx->mb, 0, _x, x); CE;
  ierr = MB_GetFldWithVec(ctx->mb, 0, _r, r); CE;
  ierr = %s(ctx, x, r); CE;
  mrc_fld_destroy(x);
  mrc_fld_destroy(r);
  pfr;
}
""" % (self._name, self._ctx_type, self._ctx_type, self._name))

    # ==================================================================
    # createJacobian1
    # ==================================================================

    # ------------------------------------------------------------------
    def findStencil1(self, exprs, x):
        # x is the state vector we need to find the stencil wrt
        # Find complete stencil ourselves
        stencil = []
        for jm in xrange(self._bs):
            f = exprs[jm]
            dep_vars = set()
            for expr in postorder_traversal(f):
                if type(expr) == F3 and expr.str == x.str:
                    dep_vars.add(expr)

            for expr in dep_vars:
                stencil.append([jm, expr.m, expr.idx])

        return stencil

    # ------------------------------------------------------------------
    def createJacobianLocal(self, file, funcname, dir, idx, x, btype=None):

        exprs = self.getBackSubstExprs(dir, idx, btype)
        stencil = self.findStencil1(exprs, x)

        file.write("static int\n")
        file.write('__attribute__((optimize("no-var-tracking-assignments")))\n')
        if dir == -1:
            suffix = "local_l_%d_%s" % (idx, btype)
        elif dir == +1:
            suffix = "local_r_%d_%s" % (idx, btype)
        else:
            suffix = "local"

        file.write("%s_%s(%s ctx, int patch, Mat J, struct mat_create_ctx *mc)\n" %
                   (funcname, suffix, self._ctx_type))

        # FIXME, the loop over patches should be outside
        file.write("""
{
  int ierr;
  struct mrc_domain *mb = ctx->mb;  // FIXME, pass it or the vector
  struct mrc_patch_info info;
  mrc_domain_get_local_patch_info(mb, patch, &info);

  pfb;
""")
        self.genLoopBegin(file, dir, idx, btype)

        file.write("""
      int jg = mrc_3d_matrix_find_global_index(mb, patch, jx, jy, jz);
""")

        for s in stencil:
            jm, im, ix = s
            file.write("ierr = __mrc_3d_matrix_set_value(mb, J, %d, %d, jg, info.global_patch, %d, jx+%d, jy+%d, jz+%d, 0., mc); CE;\n"
                       % (self._bs, jm, im, ix[0], ix[1], ix[2]))

        self.genLoopEnd(file, dir, idx)

        file.write("""
  pfr;
}
""")

    # ------------------------------------------------------------------
    def createJacobian1(self, funcname, x, file, bc):
        """generates code for the jacobian evaluation"""

        self.createJacobianLocal(file, funcname, 0, (bc.l[0], bc.r[0]), x)
        for btype in bc.btypes:
            for i in range(0, bc.l[0]):
                self.createJacobianLocal(file, funcname, -1, i, x, btype)
            for i in reversed(range(0, bc.r[0])):
                self.createJacobianLocal(file, funcname, +1, i, x, btype)

        file.write("""
static int
__attribute__((optimize("no-var-tracking-assignments")))
%s(%s ctx, struct mrc_fld *xg, Mat *pJ)
{
  struct mrc_domain *mb = xg->_domain;
  int ierr;
  struct mat_create_ctx mc;
  int nr_comps = mrc_fld_nr_comps(xg);  

  pfb;

  int nr_local_patches, nr_global_patches;
  mrc_domain_get_patches(mb, &nr_local_patches);
        mrc_domain_get_nr_global_patches(mb, &nr_global_patches);
  struct mrc_patch_info info;
  mrc_domain_get_global_patch_info(mb, 0, &info);
  for (mc.prealloc = 0; mc.prealloc <= 1; mc.prealloc++) {
    int global_size = info.ldims[0] 
                     * info.ldims[1] 
                     * info.ldims[2] 
                     * nr_global_patches;
    int local_size = info.ldims[0] 
                     * info.ldims[1] 
                     * info.ldims[2] 
                     * nr_local_patches;

    ierr = __MatCreate(mrc_domain_comm(mb),
      nr_comps*local_size, nr_comps*local_size,
      nr_comps*global_size, nr_comps*global_size,
      pJ, &mc); CE;
""" % (funcname, self._ctx_type))

        file.write("  \n")
        file.write("  ;\n")
        file.write("  for(int patch=0; patch < nr_local_patches; patch++) {\n")
        file.write("    struct mrc_patch_info info;\n")
        file.write("    mrc_domain_get_local_patch_info(mb, patch, &info);\n")
        for btype in bc.btypes:
            file.write(
                "      if (info.p_pface[FACE_LEFT].pf_btype == %s) {\n" % btype)
            for i in range(0, bc.l[0]):
                file.write("        ierr = %s_local_l_%d_%s(ctx, patch, *pJ, &mc); CE;\n"
                           % (funcname, i, btype))
            file.write("      }\n")

        file.write("      ierr = %s_local(ctx, patch, *pJ, &mc); CE;\n"
                   % (funcname))

        for btype in bc.btypes:
            file.write(
                "      if (info.p_pface[FACE_RIGHT].pf_btype == %s) {\n" % btype)
            for i in reversed(range(0, bc.r[0])):
                file.write("        ierr = %s_local_r_%d_%s(ctx, patch, *pJ, &mc); CE;\n"
                           % (funcname, i, btype))
            file.write("      }\n")

        file.write("      }\n")
        file.write("""
  }
  ierr = MatAssemblyBegin(*pJ, MAT_FINAL_ASSEMBLY); CE;
  ierr = MatAssemblyEnd  (*pJ, MAT_FINAL_ASSEMBLY); CE;

  pfr;
}

""")

    # ==================================================================
    # calcJacobian1
    # ==================================================================

    # ------------------------------------------------------------------
    def calcJacobianLocal(self, file, funcname, dir, idx, x, btype=None):

        exprs = self.getBackSubstExprs(dir, idx, btype)
        stencil = self.findStencil1(exprs, x)

        element_list = []
        for s in stencil:
            jm, im, ix = s
            f = exprs[jm]
            xx = x[im].at(ix)
            element_list.append(diff(f, xx))

        file.write("static int\n")
        file.write('__attribute__((optimize("no-var-tracking-assignments")))\n')
        if dir == -1:
            suffix = "local_l_%d_%s" % (idx, btype)
        elif dir == +1:
            suffix = "local_r_%d_%s" % (idx, btype)
        else:
            suffix = "local"

        file.write("%s_%s(%s ctx, int patch, struct mrc_fld *x, Mat J)\n" %
                   (funcname, suffix, self._ctx_type))

        file.write("""
{
  int ierr;
  struct mrc_domain *mb = x->_domain;
  struct mrc_crds *crds = mrc_domain_get_crds(mb);
  struct mrc_trafo *trafo = ctx->trafo; // FIXME: Not general
  struct mrc_patch_info info;
  mrc_domain_get_local_patch_info(mb, patch, &info);

  pfb;
""")
        self.genLoopBegin(file, dir, idx, btype)

        file.write("""
        int jg = mrc_3d_matrix_find_global_index(mb, patch, jx, jy, jz);
""")

        for i, expr in enumerate(element_list):
            jm, im, ix = stencil[i]
            file.write('double _a%d = %s;\n' % (i, gen_C(expr)))
            file.write('ierr = mrc_3d_matrix_set_value(mb, J, %d, %d, jg, info.global_patch, %d, jx+%d, jy+%d, jz+%d, _a%d); CE;\n'
                       % (self._bs, jm, im, ix[0], ix[1], ix[2], i))

        self.genLoopEnd(file, dir, idx)

        file.write("""
  pfr;
}
""")

    # ------------------------------------------------------------------
    def calcJacobian1(self, funcname, x, file, bc):
        """generates code for the jacobian evaluation"""

        self.calcJacobianLocal(file, funcname, 0, (bc.l[0], bc.r[0]), x)
        for btype in bc.btypes:
            for i in range(0, bc.l[0]):
                self.calcJacobianLocal(file, funcname, -1, i, x, btype)
            for i in reversed(range(0, bc.r[0])):
                self.calcJacobianLocal(file, funcname, +1, i, x, btype)

        file.write("""
static int
__attribute__((optimize("no-var-tracking-assignments")))
%s(%s ctx, struct mrc_fld *x, Mat J)
{
  struct mrc_domain *mb = x->_domain;
  int ierr;
  struct mrc_crds *crds = mrc_domain_get_crds(mb);
  struct mrc_trafo *trafo = ctx->trafo; // FIXME: Not general
  pfb;
  ierr = MatZeroEntries(J); CE; // FIXME? probably necessary because ADD_VALUES
""" % (funcname, self._ctx_type))

        file.write("  int nr_local_patches;\n")
        file.write("  mrc_domain_get_patches(mb, &nr_local_patches);\n")
        file.write("  for(int patch=0; patch < nr_local_patches; patch++) {\n")
        file.write("    struct mrc_patch_info info;\n")
        file.write("    mrc_domain_get_local_patch_info(mb, patch, &info);\n")
        for btype in bc.btypes:
            file.write(
                "     if (info.p_pface[FACE_LEFT].pf_btype == %s) {\n" % btype)
            for i in range(0, bc.l[0]):
                file.write("      ierr = %s_local_l_%d_%s(ctx, patch, x, J); CE;\n"
                           % (funcname, i, btype))
            file.write("    }\n")

        file.write("    ierr = %s_local(ctx, patch, x, J); CE;\n"
                   % (funcname))

        for btype in bc.btypes:
            file.write(
                "    if (info.p_pface[FACE_RIGHT].pf_btype == %s) {\n" % btype)
            for i in reversed(range(0, bc.r[0])):
                file.write("    ierr = %s_local_r_%d_%s(ctx, patch, x, J); CE;\n"
                           % (funcname, i, btype))
            file.write("    }\n")

        file.write("  }")

        file.write("""
  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY); CE;
  ierr = MatAssemblyEnd  (J, MAT_FINAL_ASSEMBLY); CE;

  pfr;
}

""")

    # ==================================================================

    # ------------------------------------------------------------------
    def genAux(self, file):
        """generates code to first evaluate auxiliary quantities"""

        if not self._aux:
            return

        for aux in self._aux:
            s_aux, expr_aux, bc_aux = aux
            if bc_aux:
                bc_aux = bc_aux
            else:
                bc_aux = "NULL"
            file.write("""
  struct mrc_fld *%sg = mrc_fld_create(mrc_domain_comm(mb));
  ierr = MB_GetGlobalFld(mb, %d, %sg); CE;
""" % ( s_aux, len(expr_aux), s_aux))

            file.write("""
  mrc_fld_foreach_patch(%sg, patch) {
    mrc_fld_foreach(%sg, jx,jy,jz, 0,0) {
            """ % ( s_aux, s_aux))

            sub, expr_list = cse([f.at((0, 0, 0)) for f in expr_aux],
                                 cse_main.numbered_symbols('__cse_'))

            for expr in sub:
                file.write("double %s = %s;\n" % (expr[0], gen_C(expr[1])))

            file.write("\n")

            for i in xrange(len(expr_list)):
                file.write('MRC_D5(%sg,%d, jx,jy,jz,patch) = %s;\n\n'
                           % (s_aux, i, gen_C(expr_list[i])))

            file.write("""
     } mrc_fld_foreach_end;
   }
""")
            file.write("""
  struct mrc_fld *%s = mrc_fld_create(mrc_domain_comm(mb));
  MB_GetLocalFld(%sg->_domain, mrc_fld_nr_comps(%sg), %s);
  M3d_FillGhostCells(ctx->bnd, %sg, %s, \"%s\");
  mrc_fld_destroy(%sg);
""" % ( s_aux, s_aux, s_aux, s_aux, s_aux, s_aux, bc_aux, s_aux))

    # ------------------------------------------------------------------
    def genAuxFinish(self, file):
        """generates code to release aux mem"""

        if not self._aux:
            return

        for aux in self._aux:  # FIXME horrible
            file.write("""
   mrc_fld_destroy(%s);
""" % (aux[0]))

    # ------------------------------------------------------------------
    def function(self, file):
        """generates code for the function evaluation"""

        file.write("""
static int
__attribute__((optimize("no-var-tracking-assignments")))
%s_local(%s ctx, struct mrc_fld *x, struct mrc_fld *f)
{
  struct mrc_domain *mb = x->_domain;
  struct mrc_crds *crds = mrc_domain_get_crds(mb);
  struct mrc_trafo *trafo = ctx->trafo; // FIXME: Not general
  pfb;
""" % (self._name, self._ctx_type))

        if self._aux or self._sub_src:
            file.write("""
  int ierr;
""")

        self.genAux(file)

        file.write("""
  mrc_fld_foreach_patch(f, patch) {
    mrc_fld_foreach(f, jx,jy,jz, 0,0) {
""")

        if False:  # FIXME, cylindrical only
            file.write("""
      struct mrc_patch_info info;  
      mrc_domain_get_local_patch_info(mb, patch, &info);
      double __attribute__((unused)) sp_zero = (jx + info->p_ix[0] == 0 ? 0. : 1.);
""")

        sub, expr_list = cse([f.at((0, 0, 0)) for f in self._func],
                             cse_main.numbered_symbols('__cse_'))

        for expr in sub:
            file.write("double %s = %s;\n" % (expr[0], gen_C(expr[1])))

        file.write("\n")

        for i in xrange(len(expr_list)):
            file.write('MRC_D5(f,%d, jx,jy,jz,patch) = %s;\n\n' %
                       (i, gen_C(expr_list[i])))

        file.write("""
     } mrc_fld_foreach_end;
   }

 """)
        self.genAuxFinish(file)

        file.write("""

   pfr;
 }
 """)

        # wrapper to generate local version first

        if self._bc:
            bc = self._bc
        else:
            bc = "NULL"

        file.write("""
static int
__attribute__((optimize("no-var-tracking-assignments")))
%s(%s ctx, struct mrc_fld *xg, struct mrc_fld *f)
{
  int ierr;

  pfb;
  assert(mrc_fld_nr_comps(f) == %d);
  struct mrc_fld *x = mrc_fld_create(mrc_domain_comm(ctx->mb));
  MB_GetFld(xg->_domain, mrc_fld_nr_comps(xg), %d, x);
  M3d_FillGhostCells(ctx->bnd, xg, x, \"%s\");
  ierr = %s_local(ctx, x, f); CE;
""" % (self._name, self._ctx_type, self._bs, self._sw, bc, self._name))

        if self._sub_src:
            file.write("""
   if (ctx->src) {
    if (mrc_fld_is_setup(ctx->src)) {
      void (*fld_axpy)(struct mrc_fld *, float, struct mrc_fld *);
      fld_axpy = (void (*)(struct mrc_fld *, float, struct mrc_fld *)) mrc_fld_get_method(f, "axpy");
      fld_axpy(f, -1., ctx->src);
    }
   }
""")

        file.write("""
  mrc_fld_destroy(x);
  pfr;
}

""")

    def findStencil(self):
        # Find complete stencil ourselves
        stencil = []
        for jm in xrange(self._bs):
            f = self._func[jm].at((0, 0, 0))
            dep_vars = set()
            for expr in postorder_traversal(f):
                if type(expr) == F3:
                    dep_vars.add(expr)

            for expr in dep_vars:
                stencil.append([jm, expr.m, expr.idx])

        return stencil

    def createJacobian(self, funcname, x, file, stencil=None):
        """generates code for the jacobian evaluation
           Doesn't seem to be used anymore..."""

        if self._bc:
            bc = self._bc
        else:
            bc = "NULL"

        if not stencil:
            stencil = self.findStencil()

        file.write("""
static int
__attribute__((optimize("no-var-tracking-assignments")))
%s(%s ctx, struct mrc_fld *xg, Mat *pJ)
{
  struct mrc_domain *mb = xg->_domain;
  struct mrc_domain_mb *sub = mrc_domain_mb(mb);
  int ierr;
  struct mat_create_ctx mc;

  int nr_comps = mrc_fld_nr_comps(xg);
        
  pfb;
  for (mc.prealloc = 0; mc.prealloc <= 1; mc.prealloc++) {
    ierr = __MatCreate(mrc_domain_comm(mb),
      nr_comps*sub->mb_loc_N[0], nr_comps*sub->mb_loc_N[0],
      nr_comps*sub->mb_N[0], nr_comps*sub->mb_N[0],
      pJ, &mc); CE;

    mrc_fld_foreach_patch(xg, patch) {
      mrc_fld_foreach(xg, jx,jy,jz, 0,0) {
            int jg = mrc_3d_matrix_find_global_index(mb, patch, jx, jy, jz);
            struct mrc_patch_info info;
            mrc_domain_get_local_patch_info(mb, patch, &info);
""" % (funcname, self._ctx_type))

        for s in stencil:
            jm, im, ix = s
            file.write('ierr = __mrc_3d_matrix_set_value(mb, *pJ, %d, %d, jg, info.global_patch, %d, jx+%d, jy+%d, jz+%d, 0., &mc); CE;\n' %
                       (self._bs, jm, im, ix[0], ix[1], ix[2]))

        file.write("""
      } mrc_fld_foreach_end;
    }
  }
  ierr = MatAssemblyBegin(*pJ, MAT_FINAL_ASSEMBLY); CE;
  ierr = MatAssemblyEnd  (*pJ, MAT_FINAL_ASSEMBLY); CE;

  pfr;
}

""")

    def calcJacobian(self, funcname, x, file, stencil=None):
        """generates code for the jacobian evaluation"""

        if self._bc:
            bc = self._bc
        else:
            bc = "NULL"

        if not stencil:
            stencil = self.findStencil()

        element_list = []
        for s in stencil:
            jm, im, ix = s
            f = self._func[jm].at((0, 0, 0))
            xx = x[im].at(ix)
            element_list.append(diff(f, xx))

        sub, expr_list = cse(element_list, cse_main.numbered_symbols('__cse_'))

        file.write("""
static int
__attribute__((optimize("no-var-tracking-assignments")))
%s(%s ctx, struct mrc_fld *x, Mat J)
{
  struct mrc_domain *mb = x->_domain;
  struct mrc_crds *crds = mrc_domain_get_crds(mb);
  struct mrc_trafo *trafo = ctx->trafo; // FIXME: not general
 
  int ierr;

  pfb;
  ierr = MatZeroEntries(J); CE; // FIXME?

  mrc_fld_foreach_patch(x, patch) {
    mrc_fld_foreach(x, jx,jy,jz, 0,0) {
      int jg = mrc_3d_matrix_find_global_index(mb, patch, jx, jy, jz);
          struct mrc_patch_info info;
          mrc_domain_get_local_patch_info(mb, patch, &info);
""" % (funcname, self._ctx_type))

        for expr in sub:
            file.write("double %s = %s;\n" % (expr[0], gen_C(expr[1])))
        file.write("\n")

        for i in xrange(len(expr_list)):
            jm, im, ix = stencil[i]
            file.write('double _a%d = %s;\n' % (i, gen_C(expr_list[i])))
            file.write('ierr = mrc_3d_matrix_set_value(mb, J, %d, %d, jg, info.global_patch, %d, jx+%d, jy+%d, jz+%d, _a%d); CE;\n' %
                       (self._bs, jm, im, ix[0], ix[1], ix[2], i))

        file.write("""
    } mrc_fld_foreach_end;
  }

  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY); CE;
  ierr = MatAssemblyEnd  (J, MAT_FINAL_ASSEMBLY); CE;

  pfr;
}

""")

    def snesJacobian(self, funcname, file):
        """generates code for snes jacobian evaluation"""

        file.write("""

static int
__attribute__((optimize("no-var-tracking-assignments")))
snes_%s(SNES snes, Vec _x, Mat *J, Mat *Jpc, 
         MatStructure *flag, void *_ctx)
{
  struct my_ctx *ctx = _ctx;
  int ierr;

  pfb;
  struct mrc_fld *xg = mrc_fld_create(mrc_domain_comm(ctx->mb));
  ierr = MB_GetFldWithVec(ctx->mb, 0, _x, xg); CE;
  struct mrc_fld *x = mrc_fld_create(mrc_domain_comm(ctx->mb));
  MB_GetLocalFld(xg->_domain, mrc_fld_nr_comps(xg), x);
  M3d_FillGhostCells(ctx->bnd, xg, x, \"%s\");

  ierr = %s(x, *Jpc); CE;

  mrc_fld_destroy(x);
  mrc_fld_destroy(xg);
  pfr;
}

""" % (funcname, funcname))
