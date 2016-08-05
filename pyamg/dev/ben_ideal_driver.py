# from pyamg.util.utils import scale_T, get_Cpt_params
# from pyamg.strength import classical_strength_of_connection, symmetric_strength_of_connection, evolution_strength_of_connection, energy_based_strength_of_connection
# from pyamg.aggregation.aggregate import standard_aggregation
# from pyamg.aggregation.tentative import pseudo_interpolation, fit_candidates, new_ideal_interpolation
# from pyamg.aggregation.smooth import jacobi_prolongation_smoother, richardson_prolongation_smoother, energy_prolongation_smoother
# from pyamg.util.utils import relaxation_as_linear_operator



import pdb, cProfile, pstats

import time
import math
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery.stencil import stencil_grid
from pyamg.aggregation.ben_ideal import ben_ideal_solver
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.util.utils import symmetric_rescaling
from pyamg.gallery import poisson


from scipy import sparse
from scipy.sparse import csr_matrix

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# General multilevel parameters
# -----------------------------
max_levels 		   = 20			# Max levels in hierarchy
max_coarse 		   = 20 		# Max points allowed on coarse grid
tol 			   = 1e-8		# Residual convergence tolerance
is_pdef 		   = True		# Assume matrix positive definite (only for aSA)
keep_levels 	   = False		# Also store SOC, aggregation, and tentative P operators
diagonal_dominance = False		# Avoid coarsening diagonally dominant rows 
coarse_solver = 'pinv'
accel = 'cg'
keep = False

# Strength of connection 
# ----------------------
#	- symmetric, strong connection if |A[i,j]| >= theta * sqrt( |A[i,i]| * |A[j,j]| )
#		+ theta (0)
#	- classical, strong connection if |A[i,j]| >= theta * max( |A[i,k]| )
#		+ theta (0)
# 	- evolution
#		+ epsilon (4)- drop tolerance, > 1. Larger e -> denser matrix. 
#		+ k (2)- ODE num time steps, step size = 1/rho(DinvA)
#		+ block_flag (F)- True / False for block matrices
#		+ symmetrize_measure (T)- True / False, True --> Atilde = 0.5*(Atilde + Atilde.T)
#		+ proj_type (l2)- Define norm for constrained min prob, l2 or D_A
# strength = ('symmetric', {'theta': 0} )
# strength = ('symmetric', {'theta': 0} )
# strength =  ('evolution', {'k': 2, 'epsilon': 4.0, 'symmetrize_measure':True})	# 'symmetric', 'classical', 'evolution'
strength = ('classical', {'theta': 0.15} )


# Aggregation 
# -----------
#	- standard
#	- naive
#		+ Differs from standard - "Each dof is considered. If it has been aggregated
# 		  skip over. Otherwise, put dof and any unaggregated neighbors in an aggregate.
#   	  Results in possibly much higher complexities than standard aggregation." 
#	- lloyd (don't know how this works...)
#		+ ratio (0.03)- fraction of the nodes which will be seeds.
#		+ maxiter (10)- maximum number iterations to perform
#		+ distance (unit)- edge weight of graph G used in Lloyd clustering.
#		  For each C[i,j]!=0,
#	    	~ unit - G[i,j] = 1
#	        ~ abs  - G[i,j] = abs(C[i,j])
#	        ~ inv  - G[i,j] = 1.0/abs(C[i,j])
#	        ~ same - G[i,j] = C[i,j]
#	        ~ sub  - G[i,j] = C[i,j] - min(C)
# aggregate = ('standard')
# aggregate = ('pairwise', {'matchings': 3, 'algorithm': 'drake'})
aggregate = None

splitting = 'RS'
# splitting = None

# Relaxation
# ---------- 
# 	- jacobi
#		+ omega (1.0)- damping parameter.
#		+ iterations (1)- number of iterations to perform.
# 	- gauss_seidel
#		+ iterations (1)- number of iterations to perform.
#		+ sweep (forward)- direction of relaxation sweep.
#			~ forward
#			~ backward
#			~ symmetric
# 	- sor
#		+ omega - damping parameter. If omega = 1.0, SOR <--> Gauss-Seidel.
#		+ iterations (1)- number of iterations to perform.
#		+ sweep (forward)- direction of relaxation sweep.
#			~ forward
#			~ backward
#			~ symmetric
# 	- block_jacobi
#		+ omega (1.0)- damping parameter.
#		+ iterations (1)- number of relaxation iterations.
#		+ blocksize (1)- block size of bsr matrix
#		+ Dinv (None)- Array holding block diagonal inverses of A.
#		  size (numBlocks, blocksize, blocksize)
#	- block_gauss_seidel
#		+ iterations (1)- number of relaxation iterations.
#		+ sweep (forward)- direction of relaxation sweep.
#			~ forward
#			~ backward
#			~ symmetric
#		+ blocksize (1)- block size of bsr matrix
#		+ Dinv (None)- Array holding block diagonal inverses of A.
#		  size (numBlocks, blocksize, blocksize)
#
# Note, Schwarz relaxation, polynomial relaxation, Cimmino relaxation,
# Kaczmarz relaxation, indexed Gauss-Seidel, and one other variant of 
# Gauss-Seidel are also available - see relaxation.py. 
# relaxation = ('jacobi', {'omega': 2.0/3.0, 'iterations': 1} )
relaxation = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1} )
# relaxation = ('richardson', {'iterations': 1})


# Improve near null space candidates
# ----------------------------------
improve_candidates = [('gauss_seidel', {'sweep': 'symmetric', 'iterations': 4})]
# improve_candidates = [('jacobi', {'omega': 2.0/3.0, 'iterations': 4})]
# improve_candidates = ('richardson', {'omega': 3.0/2.0, 'iterations': 4} )

sparsity = {'deg': 1, 'prefilter': {'k':2 } }

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Problem parameters and variables
# ---------------------------------

rand_guess 	= True
zero_rhs 	= True
problem_dim = 2
N 			= 250
epsilon 	= 0.001				# 'Strength' of aniostropy (only for 2d)
theta 		= 0.0*math.pi/16.0	# Angle of anisotropy (only for 2d)

# Empty arrays to store residuals
sa_residuals = []
asa_residuals = []
new_asa_residuals = []

# 1d Poisson 
if problem_dim == 1:
	grid_dims = [N,1]
	A = poisson((N,), format='csr')
# 2d Poisson
elif problem_dim == 2:
	grid_dims = [N,N]
	stencil = diffusion_stencil_2d(epsilon,theta)
	A = stencil_grid(stencil, grid_dims, format='csr')

# # Vectors and additional variables
[d,d,A] = symmetric_rescaling(A)
vec_size = np.prod(grid_dims)

# Zero right hand side or sin(pi x)
if zero_rhs:
	b = np.zeros((vec_size,1))
	# If zero rhs and zero initial guess, throw error
	if not rand_guess:
		print "Zero rhs and zero initial guess converges trivially."
# Note, this vector probably doesn't make sense in 2d... 
else: 
	b = np.sin(math.pi*np.arange(0,vec_size)/(vec_size-1.0))
	# b = np.array([np.sin(math.pi*np.arange(0,vec_size)/(vec_size-1.0)),np.ones(vec_size)]).T
	

# Random vs. zero initial guess
if rand_guess:
	x0 = np.random.rand(vec_size,1)
else:
	x0 = np.zeros(vec_size,1)

nii_residuals = []
rn_residuals = []
sa_residuals = []


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #


# Classical root node solver
# --------------------------

# Form classical root node multilevel solver object
start = time.clock()
ml_rn = ben_ideal_solver(A, B=None, strength=strength, aggregate=aggregate, sparsity=sparsity,
						 splitting=splitting, max_levels=max_levels, max_coarse=max_coarse,
						 presmoother=relaxation, postsmoother=relaxation,
						 improve_candidates=improve_candidates, keep=keep,
						 setup_complexity=True )

sol = ml_rn.solve(b, x0, tol, residuals=rn_residuals, accel=accel)
setup = ml_rn.setup_complexity()
cycle = ml_rn.cycle_complexity()


end = time.clock()
nii_time = end-start
rn_conv_factors = np.zeros((len(rn_residuals)-1,1))
for i in range(0,len(rn_residuals)-1):
	rn_conv_factors[i] = rn_residuals[i]/rn_residuals[i-1]

CF = np.mean(rn_conv_factors[1:])
print "Root node - ", nii_time, " seconds"
print " CF - ",CF
print " Setup complexity - ",setup
print " Cycle complexity - ",cycle
print " Effectve CF - ", CF**(1.0/cycle)


pdb.set_trace()


