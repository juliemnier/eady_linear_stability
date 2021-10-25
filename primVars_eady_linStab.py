# we the linearized Eady problem:
#
# DD PHI = sin(alpha z) on z in [0,1] with PHI'(1) = 0 and PHI'(0) = lambda PHI(0)
#
# the analytic solution is 
#    PHI = -sin(alpha z) / alpha**2 + a z + b
#
# Our approach employs the change of variable : PHI = f . phi 
# We define f = z**2 - 2z - 2/lambda so that the BC on phi are now standard Neuman.
#
# New variable phi obeys the following ODE:
#
#   f DD phi + 2f' D phi + f'' phi = sin(alpha z)
#
#
# This path is general but a little hairy here. Alternatively, we write:
#
#    PHI = I2 sin(alpha z) + c0 + c1 z   (where c0,c1 are arbitrary constants)
#  f phi = I2 sin(alpha z) + c0 + c1 z   (where c0,c1 are arbitrary constants)
#
# Introduce the galerkin decomposition of phi: phi = S gal
# Remove the two lowest degree equations with projection matrix R2
#
#    R2 f S gal = R2 I2 sin(alpha z)
#
#

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
plt.close('all')

# parameters
NZ = 256
shear = 0.5
nuh  = 0.0001
nuz  = 0.000001
nuhb = 0.0001
nuzb = 0.000001
f = 1.  
friction = 0.0001

# wn we consider:
kx = 1. 
ky = 0. 
px = 1.j*kx
py = 1.j*ky
del2h = px**2 + py**2


# define our domain
gap = 1.
center = 0.5
hTop = center + gap/2.
hBot = center - gap/2.



# define stencil matrices for imposing Neuman B.C.
nelems, ncol, nrow = chebyshev_galerkin_stencil_shape ( NZ, 21)
dat = np.empty((nelems), dtype=np.float_)
row = np.empty((nelems), dtype=np.int32 )
col = np.empty((nelems), dtype=np.int32 )
dat, col, row = chebyshev_galerkin_stencil( NZ, 21, nelems)
S_UV = sp.coo_matrix((dat, (row-1, col-1)) )
del dat, col, row # a bit of cleaning...

# define stencil matrices for imposing Neuman B.C.
nelems, ncol, nrow = chebyshev_galerkin_stencil_shape ( NZ, 20)
dat = np.empty((nelems), dtype=np.float_)
row = np.empty((nelems), dtype=np.int32 )
col = np.empty((nelems), dtype=np.int32 )
dat, col, row = chebyshev_galerkin_stencil( NZ, 20, nelems)
S_W = sp.coo_matrix((dat, (row-1, col-1)) )
del dat, col, row # a bit of cleaning...


# define stencil matrices for imposing Neuman B.C.
nelems, ncol, nrow = chebyshev_galerkin_stencil_shape ( NZ, 21)
dat = np.empty((nelems), dtype=np.float_)
row = np.empty((nelems), dtype=np.int32 )
col = np.empty((nelems), dtype=np.int32 )
dat, col, row = chebyshev_galerkin_stencil( NZ, 21, nelems)
S_theta = sp.coo_matrix((dat, (row-1, col-1)) )
del dat, col, row # a bit of cleaning...

# identity matrix
cooId = sp.coo_matrix(np.eye(NZ)) 

# define  matrix that represents multiplication by z
CEM = ct.chebyshev_elementary_multiplication(NZ, gap, center)
# define  matrix that represents integration w.r.t. z
CEI = ct.chebyshev_elementary_integration   (NZ, gap, center)
CEI.data[1][1] = 0.


# projection onto the NZ-order highest coefficients
# (discard integration constants introduced by Q.I. technique)
order = 2 # for psi and theta
dat = np.empty((NZ - order), dtype=np.float_)
row = np.empty((NZ - order), dtype=np.int32 )
col = np.empty((NZ - order), dtype=np.int32 )
dat, col, row = high_degree_projection_mat( NZ, order)
R2 = sp.coo_matrix((dat, (row-1, col-1)) )
del dat, col, row # a bit of cleaning...
order = 1 # for phi
dat = np.empty((NZ - order), dtype=np.float_)
row = np.empty((NZ - order), dtype=np.int32 )
col = np.empty((NZ - order), dtype=np.int32 )
dat, col, row = high_degree_projection_mat( NZ, order)
R1 = sp.coo_matrix((dat, (row-1, col-1)) )
del dat, col, row # a bit of cleaning...
projector = [R2,R2,R1,R1,R2,cooId]

CVS = [] # Change of variable and stencil mats
# for U,V
Change_of_vars_mat = friction * CEM.dot(CEM) + 2.*friction*hTop*CEM \
                   + (2.*(hBot - hTop) - friction* hBot**2 +2.*friction
                       *hBot*hTop)*cooId 
CVS.append(Change_of_vars_mat.dot(S_UV))
CVS.append(Change_of_vars_mat.dot(S_UV))
# for W  
CVS.append(S_W)
# for zeta = DW
CVS.append(cooId)
# for theta
Change_of_vars_mat = cooId 
CVS.append(Change_of_vars_mat.dot(S_theta))
# for pressure  
CVS.append(cooId)

# shape of the system:
eqn_sta = np.array([0, NZ-2, NZ - 2 + NZ - 2,
                             NZ - 2 + NZ - 2 + NZ - 1,
                             NZ - 2 + NZ - 2 + NZ - 1 + NZ - 1,
                             NZ - 2 + NZ - 2 + NZ - 1 + NZ - 1 + NZ - 2], dtype=np.int32)
eqn_num = np.array([   NZ-2, NZ - 2,  NZ - 1, NZ -1 , NZ -2, NZ], dtype=np.int32)
eqn_end = eqn_sta + eqn_num
var_sta = np.array([0, NZ-2, NZ - 2 + NZ - 2,
                             NZ - 2 + NZ - 2 + NZ - 2,
                             NZ - 2 + NZ - 2 + NZ - 2 + NZ - 0,
                             NZ - 2 + NZ - 2 + NZ - 2 + NZ - 0 + NZ - 2], dtype=np.int32)
var_num = np.array([   NZ-2, NZ - 2,  NZ - 2, NZ -0 , NZ -2, NZ-0], dtype=np.int32)
var_end = var_sta + var_num

ieqn = {"U" : 0, 
        "V" : 1, 
        "W" : 2,
        "zeta" : 3,
        "theta" : 4,
        "divU" : 5}

jvar = {"U" : 0, 
        "V" : 1, 
        "W" : 2,
        "zeta" : 3,
        "theta" : 4,
        "p" : 5}

def add_block_to_right_spot(large_mat, eqn_str, var_str, int_order, mul_order, zsca):
   additional_block = projector[ ieqn[ eqn_str]].dot( 
                              (CEI**int_order).dot(
                                 (CEM**mul_order).dot( 
                                    CVS[ jvar[ var_str]] 
                                                      ) 
                                                   )
                                                 ) * zsca
   additional_block = additional_block.tocoo()
   # now we concatenate:
   row_shift = eqn_sta[ ieqn [eqn_str]]
   col_shift = var_sta[ jvar [var_str]]
   nelems = additional_block.getnnz() + large_mat.getnnz()
   rRows = np.empty((nelems,), dtype=np.int32)
   rCols = np.empty((nelems,), dtype=np.int32)
   rData = np.empty((nelems,), dtype=np.complex_)
   rRows[:large_mat.getnnz()] = large_mat.row
   rCols[:large_mat.getnnz()] = large_mat.col
   rData[:large_mat.getnnz()] = large_mat.data
   rRows[large_mat.getnnz():] = additional_block.row + row_shift
   rCols[large_mat.getnnz():] = additional_block.col + col_shift
   rData[large_mat.getnnz():] = additional_block.data
   return  sp.coo_matrix((rData, (rRows, rCols)), shape = large_mat.get_shape() )

z1 = 1.+0.j

M = sp.coo_matrix((eqn_end[-1], var_end[-1]), dtype= np.complex_)
L = sp.coo_matrix((eqn_end[-1], var_end[-1]), dtype= np.complex_)

# build mass matrix
M = add_block_to_right_spot(M, "U",   "U",   2, 0, 1.+0.j)
M = add_block_to_right_spot(M, "V",   "V",   2, 0, 1.+0.j)
M = add_block_to_right_spot(M, "W",   "W",   2, 0, 1.+0.j)
M = add_block_to_right_spot(M, "theta",   "theta",   2, 0, 1.+0.j)
# build stiffness matrix
# ============   U eqn   =================================
# ... viscous term
L = add_block_to_right_spot(L, "U",   "U",   2, 0, nuh*del2h**2)
L = add_block_to_right_spot(L, "U",   "U",   0, 0, nuz)
# ... Pressure
L = add_block_to_right_spot(L, "U",   "p",   2, 0, -px)
# ... Coriolis term
L = add_block_to_right_spot(L, "U",   "V",   2, 0, f)
# ... Shear term
L = add_block_to_right_spot(L, "U",   "W",   2, 0, -shear)
L = add_block_to_right_spot(L, "U",   "U",   2, 1, -shear*px)
# ============   V eqn   =================================
# ... viscous term
L = add_block_to_right_spot(L, "V",   "V",   2, 0, nuh*del2h**2)
L = add_block_to_right_spot(L, "V",   "V",   0, 0, nuz)
# ... Pressure
L = add_block_to_right_spot(L, "V",   "p",   2, 0, -py)
# ... Coriolis term
L = add_block_to_right_spot(L, "V",   "U",   2, 0,-f)
# ... Shear term
L = add_block_to_right_spot(L, "V",   "V",   2, 1, -shear*px)
# ============   W eqn   =================================
# ... viscous term
L = add_block_to_right_spot(L, "W",   "W",   1, 0, nuh*del2h**2)
L = add_block_to_right_spot(L, "W","zeta",   0, 0, nuz)
# ... Pressure
L = add_block_to_right_spot(L, "W",   "p",   0, 0,-1. + 0.j)
# ... buoyancy term
L = add_block_to_right_spot(L, "W","theta",  1, 0, 1. + 0.j)
# ... Shear term
L = add_block_to_right_spot(L, "W",   "W",   1, 1, -shear*px)
# ===========  zeta eqn  =================================
L = add_block_to_right_spot(L, "zeta",   "W",   0, 0, 1.+0.j)         
L = add_block_to_right_spot(L, "zeta","zeta",   1, 0,-1.+0.j)         
# =========== theta eqn  =================================
# ... diffusive term
L = add_block_to_right_spot(L, "theta", "theta", 2, 0, nuhb*del2h)
L = add_block_to_right_spot(L, "theta", "theta", 0, 0, nuzb+ 0.j )
# ... shear term
L = add_block_to_right_spot(L, "theta", "theta", 2, 1, -shear * px)
# ... advection background
L = add_block_to_right_spot(L, "theta", "V",     2, 0,  shear * f )
# =========== incompressibility  =================================
L = add_block_to_right_spot(L, "divU", "U", 0, 0, px)           
L = add_block_to_right_spot(L, "divU", "V", 0, 0, py)           
L = add_block_to_right_spot(L, "divU", "zeta", 0, 0, 1.+0.j)



Ldense = L.toarray()
Mdense = M.toarray()

from scipy.linalg import eig

eigenVal, eigenVec = eig(Ldense, b=Mdense)


target_eigenval = 0.136
# PLOT THE SPECTRUM
#plt.figure()
#plt.spy(L)
plt.figure()
plt.plot(eigenVal.real, eigenVal.imag, 'o')

# PLOT THE MOST UNSTABLE MODE
plt.figure()
#mode_index = np.where(eigenVal.real == eigenVal.real.max())[0][0]
mode_index = np.argmin(np.abs(eigenVal-target_eigenval))
print ('=======')
print ('maximum growthrate s_max = '+str(eigenVal.real.max()))
print ('target:            s_usr = '+str(eigenVal[mode_index]))
print ('=======')
allVars_galn_coefs = eigenVec[:,mode_index]
U_galn_coefs = allVars_galn_coefs[var_sta[jvar["U"]]:var_end[jvar["U"]]]
V_galn_coefs = allVars_galn_coefs[var_sta[jvar["V"]]:var_end[jvar["V"]]]
W_galn_coefs = allVars_galn_coefs[var_sta[jvar["W"]]:var_end[jvar["W"]]]
theta_galn_coefs = allVars_galn_coefs[var_sta[jvar["theta"]]:var_end[jvar["theta"]]]

U_cheb_coefs = CVS[0].dot(U_galn_coefs)
V_cheb_coefs = CVS[1].dot(V_galn_coefs)
W_cheb_coefs = CVS[2].dot(W_galn_coefs)
theta_cheb_coefs = CVS[4].dot(theta_galn_coefs)

   

U_cheb_coefs[0] *= 2.
U_phys = fft.idct(U_cheb_coefs) /2.
V_cheb_coefs[0] *= 2.
V_phys = fft.idct(V_cheb_coefs) /2.
W_cheb_coefs[0] *= 2.
W_phys = fft.idct(W_cheb_coefs) /2.
theta_cheb_coefs[0] *= 2.
theta_phys = fft.idct(theta_cheb_coefs) /2.
z = np.cos( (2* np.linspace(0,NZ,num=NZ, endpoint=False) + 1.)/2./NZ*np.pi)*gap/2. + center

plt.figure()
plt.plot(U_phys.real, z)
plt.plot(U_phys.imag, z)
plt.plot(U_phys.imag**2 +
         U_phys.real**2, z)
plt.title('U')
plt.figure()
plt.plot(V_phys.real, z)
plt.plot(V_phys.imag, z)
plt.plot(V_phys.imag**2 +
         V_phys.real**2, z)
plt.title('V')
plt.figure()
plt.plot(W_phys.real, z)
plt.plot(W_phys.imag, z)
plt.plot(W_phys.imag**2 +
         W_phys.real**2, z)
plt.title('W')
plt.figure()
plt.plot(theta_phys.real, z)
plt.plot(theta_phys.imag, z)
plt.plot(theta_phys.imag**2 +
         theta_phys.real**2, z)
plt.title('theta')

def plot_wavePlan(waveProfile):
   x = np.linspace(0,4*np.pi, num=1000)
   z = center + 0.5 * gap * np.cos((2*np.linspace(0,NZ, num=NZ, endpoint=False)+1)/2./NZ*np.pi)
   X,Z= np.meshgrid(x,z)
   expX = np.exp(1.j*X)
   _,wavePlan = np.meshgrid(x,waveProfile) 
   plt.figure()
   plt.pcolormesh(x,z,(wavePlan*expX).real)
   plt.figure()
   plt.pcolormesh(x,z,(wavePlan*expX).imag)

plot_wavePlan(U_phys)
plot_wavePlan(V_phys)
plot_wavePlan(W_phys)
plot_wavePlan(theta_phys)



plt.show()

#soln_T = S.dot(soln_G)
#soln_T[0] *= 2.
#soln_z = fft.idct(soln_T)  / 2. 
#f_z = z**2 - 2.*z - 2. / LAMBDA
#soln_z = soln_z * f_z
#plt.plot(z,soln_z)
#plt.plot(z, -np.sin(alpha * z)/alpha**2 + np.cos(alpha)/alpha * z + (np.cos(alpha)/alpha - 1/alpha) / LAMBDA, 'o')
#plt.show()





