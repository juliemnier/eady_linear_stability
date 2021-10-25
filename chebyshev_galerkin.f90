

 Subroutine high_degree_projection_mat( N, remove_i_coefs, &
                     COO_dat, COO_col, COO_row)
   
   integer, intent(in) :: remove_i_coefs
   Integer, Intent(In) :: N 
   Real   (kind=8), Dimension(N - remove_i_coefs), intent(Out) :: COO_dat
   Integer(kind=4), Dimension(N - remove_i_coefs), intent(Out) :: COO_col, COO_row
   ! Internal vars.
   Integer :: j 
   
   COO_dat = 0.d0
   COO_col = 0    
   COO_row = 0

   Do j=1,N - remove_i_coefs
       COO_dat(j) = 1.d0
       COO_col(j) = j + remove_i_coefs
       COO_row(j) = j
   End do

 end subroutine


 Subroutine Chebyshev_Galerkin_stencil_shape( N, bc_type, nelems, ncol, nrow)
   integer, intent(Out) :: nelems, ncol, nrow
   Integer, Intent(In) :: N 
   Integer, Intent(In) :: bc_type
   
   nrow = N 

   !##########################################################
   ! bc_type = 00: No boundary condition
   !..........................................................
   Select Case (bc_type)
      Case(0)
         ncol  = N
         nelems = N
   !..........................................................
   ! bc_type = 10: Dirichlet at +1, f=0
   !..........................................................
      Case(10)
         ncol = N-1
         nelems = 2*(N-1)
   !..........................................................
   ! bc_type = 11: Dirichlet at -1, f=0
   !..........................................................
      Case(11)
         ncol  = N-1
         nelems = 2*(N-1)
   !..........................................................
   ! bc_type = 20: Both Dirichlet, f=0 
   !..........................................................
      Case(20)
         ncol  = N-2
         nelems = 2*(N-2)
   !..........................................................
   ! bc_type = 21: Both Neuman, Df=0
   !..........................................................
      Case(21)
         ncol  = N-2
         nelems = 2*(N-2)
   !..........................................................
   ! bc_type = 22:  Top No-Slip, Bottom Stress-Free 
   !                Top Dirichlet, Bottom Neumann   
   !..........................................................
      Case(22)
         ncol  = N-2
         nelems = 3*(N-2)
   !..........................................................
   ! bc_type = 23:  Top Stress-Free, Bottom No-Slip 
   !                Top Neumann, Bottom Dirichlet   
   !..........................................................
      Case(23)
         ncol  = N-2
         nelems = 3*(N-2)
   !..........................................................
   ! bc_type = 40: Both no-slip (f=0 and Df=0)
   !..........................................................
      Case(40)
         ncol  = N-4
         nelems = 3*(N-4)
   !..........................................................
   ! bc_type = 41: Both Stress-free (f=0 and DDf=0)
   !..........................................................
      Case(41)
         ncol  = N-4
         nelems = 3*(N-4)
   !..........................................................
   ! bc_type = 42:  Top No-Slip, Bottom Stress-Free 
   !                Top Dirichlet, Bottom Neumann   
   !..........................................................
      Case(42)
         ncol  = N-4
         nelems = 5*(N-4)
   !..........................................................
   ! bc_type = 43:  Top Stress-Free, Bottom No-Slip
   !..........................................................
      Case(43)
         ncol  = N-4
         nelems = 5*(N-4)
   end Select
   !..........................................................
   ! Done with boundary condition            
   !##########################################################

 End Subroutine Chebyshev_Galerkin_stencil_shape

 Subroutine Chebyshev_Galerkin_stencil( N, bc_type, nelems, &
                     COO_dat, COO_col, COO_row)
   
   integer, intent(in) :: nelems
   Integer, Intent(In) :: N 
   Integer, Intent(In) :: bc_type
   Real   (kind=8), Dimension(nelems), intent(Out) :: COO_dat
   Integer(kind=4), Dimension(nelems), intent(Out) :: COO_col, COO_row
   ! Internal vars.
   Integer :: j 
   Integer :: fill_in_index
   Real(8) :: naux, maux
   Integer :: nrow
   integer, parameter :: dp = 8
   
   COO_dat = 0.d0
   COO_col = 0    
   COO_row = 0

   nrow = N 
   fill_in_index = 1

   !##########################################################
   ! bc_type = 00: No boundary condition
   !..........................................................
   Select Case (bc_type)
      Case(0)
         ! build the identity
         Do j=1,N
            COO_dat(j) = 1.d0
            COO_row(j) = j
            COO_col(j) = j
         End do
   !..........................................................
   ! bc_type = 10: Dirichlet at +1, f=0
   !..........................................................
      Case(10)
            Do j = 1,N-1
               COO_dat(fill_in_index) =-1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               COO_dat(fill_in_index) = 1.0d0
               COO_row(fill_in_index) = j+1
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   !..........................................................
   ! bc_type = 11: Dirichlet at -1, f=0
   !..........................................................
      Case(11)
            Do j = 1,N-1
               COO_dat(fill_in_index) = 1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               COO_dat(fill_in_index) = 1.0d0
               COO_row(fill_in_index) = j+1
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   !..........................................................
   ! bc_type = 20: Both Dirichlet, f=0 
   !..........................................................
      Case(20)
            Do j = 1,N-2
               COO_dat(fill_in_index) =-1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               COO_dat(fill_in_index) = 1.0d0
               COO_row(fill_in_index) = j+2
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   !..........................................................
   ! bc_type = 21: Both Neuman, Df=0
   !..........................................................
      Case(21)
            Do j = 1,N-2
               COO_dat(fill_in_index) =-1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               naux = Real(j+1, kind=dp)
               COO_dat(fill_in_index) = (naux-2.0d0)**2/naux**2
               COO_row(fill_in_index) = j+2
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   !..........................................................
   ! bc_type = 22:  Top No-Slip, Bottom Stress-Free 
   !                Top Dirichlet, Bottom Neumann   
   !..........................................................
      Case(22)
            Do j = 1,N-2
               naux = Real(j-1, kind=dp)
               ! diagonal of ones
               COO_dat(fill_in_index) = 1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal 
               COO_dat(fill_in_index) = - (4.d0*naux + 4.d0)/&
                           (2.d0*naux**2 + 6.d0*naux + 5.d0)
               COO_row(fill_in_index) = j+1
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subsubdiagonal 
               COO_dat(fill_in_index) = -(naux**2 + (naux+1.d0)**2)/&
                         ((naux+1.d0)**2 + (naux+2.d0)**2)
               COO_row(fill_in_index) = j+2
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   !..........................................................
   ! bc_type = 23:  Top Stress-Free, Bottom No-Slip 
   !                Top Neumann, Bottom Dirichlet   
   !..........................................................
      Case(23)
            Do j = 1,N-2
               naux = Real(j-1, kind=dp)
               ! diagonal of ones
               COO_dat(fill_in_index) = 1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal 
               COO_dat(fill_in_index) =   (4.d0*naux + 4.d0)/&
                           (2.d0*naux**2 + 6.d0*naux + 5.d0)
               COO_row(fill_in_index) = j+1
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subsubdiagonal 
               COO_dat(fill_in_index) = -(naux**2 + (naux+1.d0)**2)/&
                         ((naux+1.d0)**2 + (naux+2.d0)**2)
               COO_row(fill_in_index) = j+2
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   !..........................................................
   ! bc_type = 40: Both no-slip (f=0 and Df=0)
   !..........................................................
      Case(40)
            Do j = 1,N-4
               COO_dat(fill_in_index) =-1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               naux = real(j+3, kind=dp)
               COO_dat(fill_in_index) = (2.d0*naux-4.d0)/(naux-1.d0)
               COO_row(fill_in_index) = j+2
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               COO_dat(fill_in_index) =-(naux-3.d0)/(naux-1.d0)
               COO_row(fill_in_index) = j+4
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   !..........................................................
   ! bc_type = 41: Both Stress-free (f=0 and DDf=0)
   !..........................................................
      Case(41)
            Do j = 1,N-4
               COO_dat(fill_in_index) =-1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               naux = Real(j+3, kind=dp)
               maux = 24.d0*naux/(2.d0*naux**2-4.d0*naux+3.d0)
               maux = maux - 18.d0/(naux-1.d0)
               COO_dat(fill_in_index) = maux +2.d0
               COO_row(fill_in_index) = j+2
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               COO_dat(fill_in_index) =-maux -1.d0
               COO_row(fill_in_index) = j+4
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   !..........................................................
   ! bc_type = 42:  Top No-Slip, Bottom Stress-Free 
   !                Top Dirichlet, Bottom Neumann   
   !..........................................................
      Case(42)
            Do j = 1,N-4
               naux = Real(j-1, kind=dp)
               ! diagonal of ones
               COO_dat(fill_in_index) = 1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal j+1
               COO_dat(fill_in_index) = - (2.d0*naux + 2.d0) /&
                        (naux**2 + 5.d0*naux + 7.d0)
               COO_row(fill_in_index) = j+1
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal j+2
               COO_dat(fill_in_index) = - ( 2.d0*naux**3 +&
                                           12.d0*naux**2 +&
                                           28.d0*naux    +&
                                           24.d0)/&
                                          (       naux**3 +&
                                            8.d0*naux**2 +&
                                           22.d0*naux    +&
                                           21.d0)
               COO_row(fill_in_index) = j+2
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal j+3
               COO_dat(fill_in_index) =   (2.d0*naux + 2.d0) /&
                        (naux**2 + 5.d0*naux + 7.d0)
               COO_row(fill_in_index) = j+3
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal j+4
               COO_dat(fill_in_index) =   (       naux**3 +&
                                            4.d0*naux**2 +&
                                            6.d0*naux    +&
                                            3.d0)/&
                                          (       naux**3 +&
                                            8.d0*naux**2 +&
                                           22.d0*naux    +&
                                           21.d0)
               COO_row(fill_in_index) = j+4
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   !..........................................................
   ! bc_type = 43:  Top Stress-Free, Bottom No-Slip
   !..........................................................
      Case(43)
            Do j = 1,N-4
               naux = Real(j-1, kind=dp)
               ! diagonal of ones
               COO_dat(fill_in_index) = 1.0d0
               COO_row(fill_in_index) =  j
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal j+1
               COO_dat(fill_in_index) =   (2.d0*naux + 2.d0) /&
                        (naux**2 + 5.d0*naux + 7.d0)
               COO_row(fill_in_index) = j+1
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal j+2
               COO_dat(fill_in_index) = - ( 2.d0*naux**3 +&
                                           12.d0*naux**2 +&
                                           28.d0*naux    +&
                                           24.d0)/&
                                          (       naux**3 +&
                                            8.d0*naux**2 +&
                                           22.d0*naux    +&
                                           21.d0)
               COO_row(fill_in_index) = j+2
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal j+3
               COO_dat(fill_in_index) = - (2.d0*naux + 2.d0) /&
                        (naux**2 + 5.d0*naux + 7.d0)
               COO_row(fill_in_index) = j+3
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
               ! subdiagonal j+4
               COO_dat(fill_in_index) =   (       naux**3 +&
                                            4.d0*naux**2 +&
                                            6.d0*naux    +&
                                            3.d0)/&
                                          (       naux**3 +&
                                            8.d0*naux**2 +&
                                           22.d0*naux    +&
                                           21.d0)
               COO_row(fill_in_index) = j+4
               COO_col(fill_in_index) =  j
               fill_in_index = fill_in_index + 1 
            End do
   End Select
   !..........................................................
   ! Done with boundary condition            
   !##########################################################
      

   

 End Subroutine Chebyshev_Galerkin_stencil
