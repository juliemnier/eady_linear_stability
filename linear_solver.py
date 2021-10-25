
# libraries we need
import numpy as np
import scipy.sparse as sp
import cheby_tools as ct
import scipy.fftpack as fft
import scipy.sparse.linalg as la
from galerkin_stencils import chebyshev_galerkin_stencil_shape
from galerkin_stencils import chebyshev_galerkin_stencil
from galerkin_stencils import high_degree_projection_mat
import matplotlib.pyplot as plt

class linearSolver():
    def __init__(self, NZ = 64, nVars = 1):
       self.NZ = NZ
       self.nVars = nVars

    def set_geometry(self, gap=1., center=0.5):
       self.gap=gap
       self.center=0.5
       self.hTop = center + gap/2.
       self.hBot = center - gap/2.

    def cheby_matrices(self):
       self._cooId = sp.coo_matrix(np.eye(self.NZ)) 
       # define  matrix that represents multiplication by z
       self._CEM = ct.chebyshev_elementary_multiplication(self.NZ, self.gap, self.center)
       # define  matrix that represents integration w.r.t. z
       self._CEI = ct.chebyshev_elementary_integration   (self.NZ, self.gap, self.center)
       self._CEI.data[1][1] = 0.

    def add_BCs_and_eqn_order(self, BC_per_variable = [2], BC_codes=[20], eqn_order=[2]):
       self.BC_codes = [x for x in BC_codes]
       self.BC_count = [x for x in BC_per_variable]
       self.eq_order = [x for x in eqn_order]
       self.order_max = 0
       for x in eqn_order:
           if (x>self.order_max):
               self.order_max = x
       self.eqn_sta = [0]
       self.eqn_num = []
       self.eqn_end = []
       for x in self.eq_order:
           self.eqn_num.append(self.NZ-x)
           self.eqn_end.append(self.eqn_sta[-1] + self.eqn_num[-1])
           self.eqn_sta.append(self.eqn_end[-1])
       self.var_sta = [0]
       self.var_num = []
       self.var_end = []
       for x in self.BC_count:
           self.var_num.append(self.NZ-x)
           self.var_end.append(self.var_sta[-1] + self.var_num[-1])
           self.var_sta.append(self.var_end[-1])

    def build_stencils(self):
       self.stencils=[]
       for bc in self.BC_codes:
         # define stencil matrices for imposing Neuman B.C.
         nelems, ncol, nrow = chebyshev_galerkin_stencil_shape ( self.NZ, bc)
         dat = np.empty((nelems), dtype=np.float_)
         row = np.empty((nelems), dtype=np.int32 )
         col = np.empty((nelems), dtype=np.int32 )
         dat, col, row = chebyshev_galerkin_stencil( self.NZ, bc, nelems)
         self.stencils.append(sp.coo_matrix((dat, (row-1, col-1)) ))
         del dat, col, row # a bit of cleaning...

    def prepare_QI_projectors(self): 
        self._R=[]
        for order in range(self.order_max+1):
            dat = np.empty((self.NZ - order), dtype=np.float_)
            row = np.empty((self.NZ - order), dtype=np.int32 )
            col = np.empty((self.NZ - order), dtype=np.int32 )
            dat, col, row = high_degree_projection_mat( self.NZ, order)
            self._R.append( sp.coo_matrix((dat, (row-1, col-1)) ) )
            del dat, col, row # a bit of cleaning...
        self.projectors=[]
        self.projectors = [self._R[x] for x in self.eq_order]

    def change_of_variable(self, list_of_mats):
        self.CVS = [list_of_mats[x].dot( self.stencils[x]) for x in range(self.nVars)]

    def provide_variable_names(self, dictionary_eqn, dictionary_var):
        self._ieqn = dictionary_eqn.copy()
        self._jvar = dictionary_var.copy()

    def initialize_operators(self):
        self.M = sp.coo_matrix((self.eqn_end[-1], self.var_end[-1]), dtype=np.complex_)
        self.L = sp.coo_matrix((self.eqn_end[-1], self.var_end[-1]), dtype=np.complex_)

    def add_linear_operator_to_mass(self, eqn_str, var_str, int_order, mul_order, zsca):
         additional_block = self.projectors[ self._ieqn[ eqn_str]].dot( 
                                    (self._CEI**int_order).dot(
                                       (self._CEM**mul_order).dot( 
                                          self.CVS[ self._jvar[ var_str]] 
                                                            ) 
                                                         )
                                                       ) * zsca
         additional_block = additional_block.tocoo()
         # now we concatenate:
         row_shift = self.eqn_sta[ self._ieqn [eqn_str]]
         col_shift = self.var_sta[ self._jvar [var_str]]
         nelems = additional_block.getnnz() + self.M.getnnz()
         rRows = np.empty((nelems,), dtype=np.int32)
         rCols = np.empty((nelems,), dtype=np.int32)
         rData = np.empty((nelems,), dtype=np.complex_)
         rRows[:self.M.getnnz()] = self.M.row
         rCols[:self.M.getnnz()] = self.M.col
         rData[:self.M.getnnz()] = self.M.data
         rRows[self.M.getnnz():] = additional_block.row + row_shift
         rCols[self.M.getnnz():] = additional_block.col + col_shift
         rData[self.M.getnnz():] = additional_block.data
         self.M = sp.coo_matrix((rData, (rRows, rCols)), shape = self.M.get_shape() )


    def add_linear_operator_to_stiffness(self, eqn_str, var_str, int_order, mul_order, zsca):
         additional_block = self.projectors[ self._ieqn[ eqn_str]].dot( 
                                    (self._CEI**int_order).dot(
                                       (self._CEM**mul_order).dot( 
                                          self.CVS[ self._jvar[ var_str]] 
                                                            ) 
                                                         )
                                                       ) * zsca
         additional_block = additional_block.tocoo()
         # now we concatenate:
         row_shift = self.eqn_sta[ self._ieqn [eqn_str]]
         col_shift = self.var_sta[ self._jvar [var_str]]
         nelems = additional_block.getnnz() + self.L.getnnz()
         rRows = np.empty((nelems,), dtype=np.int32)
         rCols = np.empty((nelems,), dtype=np.int32)
         rData = np.empty((nelems,), dtype=np.complex_)
         rRows[:self.L.getnnz()] = self.L.row
         rCols[:self.L.getnnz()] = self.L.col
         rData[:self.L.getnnz()] = self.L.data
         rRows[self.L.getnnz():] = additional_block.row + row_shift
         rCols[self.L.getnnz():] = additional_block.col + col_shift
         rData[self.L.getnnz():] = additional_block.data
         self.L = sp.coo_matrix((rData, (rRows, rCols)), shape = self.L.get_shape() )

    def dense_solve(self):
         self.Ldense = self.L.toarray()
         self.Mdense = self.M.toarray()
         from scipy.linalg import eig
         self.eigenVal, self.eigenVec = eig(self.Ldense, b= self.Mdense)


