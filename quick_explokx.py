# we the linearized Eady problem:
import numpy as np
from scipy.linalg import eig
import scipy.sparse as sp
import cheby_tools as ct
import scipy.fftpack as fft
import scipy.sparse.linalg as la
from galerkin_stencils import chebyshev_galerkin_stencil_shape
from galerkin_stencils import chebyshev_galerkin_stencil
from galerkin_stencils import high_degree_projection_mat
import matplotlib.pyplot as plt
plt.close('all')


####################################
# make it a function 

def most_unstable_kx(beta, shear, N, H, f, nuh, nuz, friction, NZ, precision=20,display=True):
    # needed function    
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

    #extended parameters
    nuzb = nuz 
    nuhb = nuh  #to change if needed
    Ld = N*H/f
    # define our domain

    gap = 1.
    center = 0.5
    hTop = center + gap/2.
    hBot = center - gap/2.


    kx = 1.6/Ld #kmax, instability greatest
    kmax = kx
    #print('====================')
    #print('kmax ='+str(kx))
    #print('====================')
    if friction<0.6:
        K = np.linspace(kx-0.1*kx,kx+0.2*kx,num=precision)
    else:
        K = np.linspace(kx-0.1*kx,kx+9*kx,num=precision+10)
    l=0.
    ky = l
    growth_rates = []
    for i in range(len(K)):
      kx = K[i]
      mu = np.sqrt(kx**2+l**2)*Ld
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
      #  ... Coriolis term
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

      eigenVal, eigenVec = eig(Ldense, b=Mdense)
      growth_rates.append(eigenVal.real.max())

    maxi=max(growth_rates)
    #print('max growth rate : '+str(maxi))
    max_index = growth_rates.index(maxi)
    print('for kx='+str(K[max_index]))

    if display==True:
        plt.figure()
        plt.plot(K,growth_rates,'o', color='black')
        plt.plot(K,0.000725*np.ones(np.shape(K)),'--',label = 'Maximum growthrate on DNS (coral)')
        plt.vlines(kmax,min(growth_rates),max(growth_rates)+0.25*max(growth_rates),color='orange',label = 'Analytical most unstable kx (eady)')
        #plt.vlines(K[max_index],min(growth_rates),0.0020,label = 'Linear stability most unstable kx')
        plt.xlabel(r'$k_x$')
        ax = plt.gca()
        ax.set_ylim([min(growth_rates)-0.1*min(growth_rates),max(growth_rates)+0.25*max(growth_rates) ])
        plt.ylabel(r'$\sigma$')
        plt.title('Maximum growth rate for DNS run 0003 (ky = 0)')
        plt.grid()
        plt.legend()
        plt.show()
    return [kmax,K[max_index],maxi] #maximum kx for eady pb associated, max kx for b plane pb, associated max growth rate 

