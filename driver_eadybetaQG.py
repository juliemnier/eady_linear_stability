# we the linearized Eady problem:
#
# Our approach employs the change of variable : PHI = f . phi 
# We define f = z**2 - 2z - 2/lambda so that the BC on phi are now standard Neuman.
#
# New variable phi obeys the following ODE:
#
#   f DD phi + 2f' D phi + f'' phi = sin(alpha z)
#
# This path is general but a little hairy here. Alternatively, we write:
#
#    PHI = I2 sin(alpha z) + c0 + c1 z   (where c0,c1 are arbitrary constants)
#  f phi = I2 sin(alpha z) + c0 + c1 z   (where c0,c1 are arbitrary constants)
#
# Introduce the galerkin decomposition of phi: phi = S gal
# Remove the two lowest degree equations with projection matrix R2
#    R2 f S gal = R2 I2 sin(alpha z)

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
Ly = 2500.
Lx = 2500.
shear = 0.5 #Rossby
nuh  = 0.#1.2, here nonviscous case
nuz  = 0.01 #0.0001 same
nuzb = 0.01 #same
nuhb = 0.#1.2, same
f = 100.  
friction = 0.00001 
N = 10.
H = 1.


# define our domain

gap = 1.
center = 0.5
hTop = center + gap/2.
hBot = center - gap/2.

####################################
####################################

## Analytical solution for single most unstable mode

# Considering:

Ld = N*H/f
beta = H*shear*0.5/Ld**2

print('=====================')
print(r'$\beta$='+str(beta))
print('=====================')

kx = 1.6/Ld #kmax, instability greatest
l = 0. 
ky= l
mu = np.sqrt(kx**2+l**2)*Ld
mum = 1.61
sigma_E = 0.31*shear*f/N
ci = sigma_E/kx
cr = 0.5*shear*H
c = np.sqrt(ci**2+cr**2)

z = np.linspace(0,H,num=NZ,endpoint=False)
theta_m = np.arctan(ci*shear*H*np.sinh(mum*z/H)/(mum*(c)**2*np.cosh(mum*z/H)-shear*cr*np.sinh(mum*z/H)))
psi_m = np.sqrt(((ci*shear*H*np.sinh(mum*z/H))/(mum*c**2))**2+ (np.cosh(mum*z/H)-(shear*cr*np.sinh(mum*z/H))/(mum*c**2))**2)


c = cr + 1.j*ci
phi_vallis = np.cosh(mum*z/H) - np.sinh(mum*z/H)*shear*H/(mum*c)


#plt.figure()
#plt.plot(theta_m,z,'--',label='phase')
#plt.plot(psi_m, z, '--',label='amplitude')
#plt.legend()
#plt.grid()
#plt.title('Analytical most unstable mode')




def plot_waveAnalytics(waveProfile,var):
   x = np.linspace(0,4*np.pi, num=1000)
   z = np.linspace(0,H,num=NZ,endpoint=False)
   X,Z= np.meshgrid(x,z)
   expX = np.exp(1.j*X)
   _,wavePlan = np.meshgrid(x,waveProfile)
   plt.figure()
   plt.pcolormesh(x,z,(wavePlan*expX).real,shading='auto')
   plt.colorbar()
   plt.xlabel('x')
   plt.ylabel('z')
   plt.title(var)
   plt.contour(X, Z,wavePlan*expX,colors='w', linewidths=1)
#plot_waveAnalytics(phi_vallis,'phi vallis')

plot_waveAnalytics(phi_vallis,r'EADY: Analytical most unstable mode, $\psi$')


###################################
###################################

# wn we consider:
px = 1.j*kx
py = 1.j*ky
del2h = px**2 + py**2


# define stencil matrices for imposing Neuman B.C.
if (friction<1.e-8):
    BCcode = 23
else:
    BCcode = 21
nelems, ncol, nrow = chebyshev_galerkin_stencil_shape ( NZ, BCcode)
dat = np.empty((nelems), dtype=np.float_)
row = np.empty((nelems), dtype=np.int32 )
col = np.empty((nelems), dtype=np.int32 )
dat, col, row = chebyshev_galerkin_stencil( NZ, BCcode, nelems)
S_psi = sp.coo_matrix((dat, (row-1, col-1)) )
del dat, col, row # a bit of cleaning...


# define stencil matrices for imposing Neuman B.C.
if (friction<1.e-8):
    BCcode = 43
else:
    BCcode = 41
nelems, ncol, nrow = chebyshev_galerkin_stencil_shape ( NZ, BCcode)
dat = np.empty((nelems), dtype=np.float_)
row = np.empty((nelems), dtype=np.int32 )
col = np.empty((nelems), dtype=np.int32 )
dat, col, row = chebyshev_galerkin_stencil( NZ, BCcode, nelems)
S_phi = sp.coo_matrix((dat, (row-1, col-1)) )
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
order = 4 # for phi
dat = np.empty((NZ - order), dtype=np.float_)
row = np.empty((NZ - order), dtype=np.int32 )
col = np.empty((NZ - order), dtype=np.int32 )
dat, col, row = high_degree_projection_mat( NZ, order)
R4 = sp.coo_matrix((dat, (row-1, col-1)) )
del dat, col, row # a bit of cleaning...
projector = [R2,R4,R2]

CVS = [] # Change of variable and stencil mats
# for psi
Change_of_vars_mat = friction * CEM.dot(CEM) + 2.*friction*hTop*CEM \
                   + (2.*(hBot - hTop) - friction* hBot**2 +2.*friction
                       *hBot*hTop)*cooId 
CVS.append(Change_of_vars_mat.dot(S_psi))
# for phi
Change_of_vars_mat = friction * CEM.dot(CEM) + 2.*friction*hTop*CEM \
                   + (4.*(hBot - hTop) - friction* hBot**2 +2.*friction
                       *hBot*hTop)*cooId 
CVS.append(Change_of_vars_mat.dot(S_phi))
# for theta
Change_of_vars_mat = cooId 
CVS.append(Change_of_vars_mat.dot(S_theta))

# shape of the system:
eqn_sta = np.array([0, NZ-2, NZ - 4 + NZ - 2], dtype=np.int32)
eqn_num = np.array([   NZ-2, NZ - 4,  NZ - 2], dtype=np.int32)
eqn_end = eqn_sta + eqn_num
var_sta = np.array([0, NZ-2, NZ - 4 + NZ - 2], dtype=np.int32)
var_num = np.array([   NZ-2, NZ - 4,  NZ - 2], dtype=np.int32)
var_end = var_sta + var_num

ieqn = {"psi" : 0, 
        "phi" : 1, 
        "theta" : 2} 
jvar = {"psi" : 0, 
        "phi" : 1, 
        "theta" : 2} 

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
M = add_block_to_right_spot(M, "psi",   "psi",   2, 0, -del2h)  
M = add_block_to_right_spot(M, "phi",   "phi",   4, 0,  del2h**2)
M = add_block_to_right_spot(M, "phi",   "phi",   2, 0,  del2h)
M = add_block_to_right_spot(M, "theta", "theta", 2, 0,  1.+0.j)
# build stiffness matrix
# psi eqn
# ... viscous term
L = add_block_to_right_spot(L, "psi",   "psi",   2, 0, -nuh*del2h**2) 
L = add_block_to_right_spot(L, "psi",   "psi",   0, 0, -nuz*del2h)
# ... Coriolis term
L = add_block_to_right_spot(L, "psi",   "phi",   1, 0, -f *del2h)
L = add_block_to_right_spot(L, "psi",   "psi",   2, 0, beta*px)
L = add_block_to_right_spot(L, "psi",   "phi",   1, 0,  -beta*py)
# ... Shear term
L = add_block_to_right_spot(L, "psi",   "phi",   2, 0, -shear*py*del2h)
L = add_block_to_right_spot(L, "psi",   "psi",   2, 1,  shear*px*del2h)

# phi eqn
# ... viscous term
L = add_block_to_right_spot(L, "phi",   "phi",   4, 0,   nuh*del2h**3)
L = add_block_to_right_spot(L, "phi",   "phi",   2, 0,   nuh*del2h**2)
L = add_block_to_right_spot(L, "phi",   "phi",   2, 0,   nuz*del2h**2)
L = add_block_to_right_spot(L, "phi",   "phi",   0, 0,   nuz*del2h)
# ... buoyancy term
L = add_block_to_right_spot(L, "phi",   "theta", 4, 0,  -del2h)
# ... Coriolis term
L = add_block_to_right_spot(L, "phi",   "psi",   3, 0,  -f *del2h)
L = add_block_to_right_spot(L, "phi",   "psi",   3, 0,  -beta*py)
L = add_block_to_right_spot(L, "phi",   "phi",   2, 0,  -beta*px)
# ... Shear term
# use II z dzz phi = z phi - 2 I phi
L = add_block_to_right_spot(L, "phi",   "phi",   4, 1,  - shear* px *del2h**2) # originally "psi","phi",..
L = add_block_to_right_spot(L, "phi",   "phi",   2, 1,  - shear* del2h * px)
L = add_block_to_right_spot(L, "phi",   "phi",   3, 0, 2.*shear* del2h * px)

# theta eqn
# ... diffusive term
L = add_block_to_right_spot(L, "theta", "theta", 2, 0, nuhb*del2h)
L = add_block_to_right_spot(L, "theta", "theta", 0, 0, nuzb+ 0.j )
# ... shear term
L = add_block_to_right_spot(L, "theta", "theta", 2, 1, -shear * px)
# ... advection background
L = add_block_to_right_spot(L, "theta", "psi",   2, 0, -shear * f  * px)
L = add_block_to_right_spot(L, "theta", "phi",   1, 0, +shear * f  * py)
# ... Background stratification
L = add_block_to_right_spot(L, "theta", "phi",   2, 0, N**2*del2h)


Ldense = L.toarray()
Mdense = M.toarray()

from scipy.linalg import eig

eigenVal, eigenVec = eig(Ldense, b=Mdense)

print ('=======')
print ('EADY maximum growthrate s_max = '+str(eigenVal.real.max()))
print ('=======')

# PLOT THE SPECTRUM
#plt.figure()
#plt.spy(L)
plt.figure()
plt.plot(eigenVal.real, eigenVal.imag, 'o')
plt.title('EADY: Spectrum with coral')
plt.xlabel(r'Re(\lambda)')
plt.ylabel(r'Im(\lambda)')
plt.grid()

# PLOT THE MOST UNSTABLE MODE
mode_index = np.where(eigenVal.real == eigenVal.real.max())[0][0]
allVars_galn_coefs = eigenVec[:,mode_index]
phi_galn_coefs = allVars_galn_coefs[var_sta[jvar["phi"]]:var_end[jvar["phi"]]]
psi_galn_coefs = allVars_galn_coefs[var_sta[jvar["psi"]]:var_end[jvar["psi"]]]
theta_galn_coefs = allVars_galn_coefs[var_sta[jvar["theta"]]:var_end[jvar["theta"]]]

psi_cheb_coefs = CVS[0].dot(psi_galn_coefs)
phi_cheb_coefs = CVS[1].dot(phi_galn_coefs)
theta_cheb_coefs = CVS[2].dot(theta_galn_coefs)

psi_cheb_coefs[0] *= 2.
psi_phys = fft.idct(psi_cheb_coefs) /2.
phi_cheb_coefs[0] *= 2.
phi_phys = fft.idct(phi_cheb_coefs) /2.
theta_cheb_coefs[0] *= 2.
theta_phys = fft.idct(theta_cheb_coefs) /2.
z = np.cos( (2* np.linspace(0,NZ,num=NZ, endpoint=False) + 1.)/2./NZ*np.pi)*gap/2. + center

psi_physcp = psi_phys/max(abs(psi_phys))
#thet = 2*np.arctan((psi_physcp.imag)/(psi_physcp.real+abs(psi_physcp)))
thet = np.arctan((psi_physcp.imag)/(psi_physcp.real))
thet=thet-thet[NZ-1]
zan = np.linspace(0,H,NZ)
# Plot Amplitude and phase for coral and analytical most unstable mode

plt.figure()
plt.plot(psi_m,zan,'--',label='analytical solution')
plt.plot(abs(psi_physcp), z,label='with coral')
plt.title(r'EADY: Amplitude $\psi$ for most unstable mode')
plt.ylabel('z')
plt.legend()
plt.grid()

plt.figure()
plt.plot(theta_m,zan,'--',label='analytical solution')
plt.plot(thet, z,label='with coral')
plt.grid()
plt.legend()
plt.ylabel('z')
plt.title(r'EADY: Phase $\psi$ for most unstable mode')


def plot_wavePlan(waveProfile,var):
   x = np.linspace(0,4*np.pi, num=1000)
   z = center + 0.5 * gap * np.cos((2*np.linspace(0,NZ, num=NZ, endpoint=False)+1)/2./NZ*np.pi)
   X,Z= np.meshgrid(x,z)
   expX = np.exp(1.j*X)
   _,wavePlan = np.meshgrid(x,waveProfile) 
   plt.figure()
   plt.pcolormesh(x,z,(wavePlan*expX).real,shading='auto')
   plt.xlabel('x')
   plt.colorbar()
   plt.ylabel('z')
   plt.title(var)
   plt.contour(X, Z,wavePlan*expX,colors='w', linewidths=1)
#plot_wavePlan(psi_m,'ampl vallis')
plot_wavePlan(psi_physcp,r'Coral most unstable mode, $\psi$')
plot_wavePlan(phi_phys,r'$\phi$')
plot_wavePlan(theta_phys,'b')

plt.show()
