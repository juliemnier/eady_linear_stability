#!/bin/bash
rm galerkin_stencils.pyf galerkin_stencils.so
f2py chebyshev_galerkin.f90 -m galerkin_stencils -h galerkin_stencils.pyf

f2py -c galerkin_stencils.pyf chebyshev_galerkin.f90 

