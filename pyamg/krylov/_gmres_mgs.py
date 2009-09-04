from numpy import array, zeros, sqrt, ravel, abs, max, arange, conjugate, hstack, isnan, isinf, iscomplex, real
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.sputils import upcast
import scipy.lib.blas as blas
from warnings import warn
from pyamg import amg_core
import scipy

__docformat__ = "restructuredtext en"

__all__ = ['gmres_mgs']


def apply_givens(Q, v, k):
    ''' 
    Apply the first k Given's rotations in Q to v 
    
    Parameters
    ----------
    Q : {list} 
        list of consecutive 2x2 Given's rotations 
    v : {array}
        vector to apply the rotations to
    k : {int}
        number of rotations to apply.

    Returns
    -------
    v is changed in place

    Notes
    -----
    This routine is specialized for GMRES.  It assumes that the first Given's
    rotation is for dofs 0 and 1, the second Given's rotation is for dofs 1 and 2,
    and so on.
    '''

    for j in xrange(k):
        Qloc = Q[j]
        v[j:j+2] = scipy.dot(Qloc, v[j:j+2])


def gmres_mgs(A, b, x0=None, tol=1e-5, restrt=None, maxiter=None, xtype=None, M=None, callback=None, residuals=None, reorth=False):
    '''
    Generalized Minimum Residual Method (GMRES)
        GMRES iteratively refines the initial solution guess to the system Ax = b
        Modified Gram-Schmidt version

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n,1)
    x0 : {array, matrix}
        initial guess, default is a vector of zeros
    tol : float
        relative convergence tolerance, i.e. tol is scaled by ||b||
    restrt : {None, int}
        - if int, restrt is max number of inner iterations
          and maxiter is the max number of outer iterations
        - if None, do not restart GMRES, and max number of inner iterations is maxiter
    maxiter : {None, int}
        - if restrt is None, maxiter is the max number of inner iterations 
          and GMRES does not restart  
        - if restrt is int, maxiter is the max number of outer iterations, 
          and restrt is the max number of inner iterations
    xtype : type
        dtype for the solution, default is automatic type detection
    M : {array, matrix, sparse matrix, LinearOperator}
        n x n, inverted preconditioner, i.e. solve M A x = b.
    callback : function
        User-supplied funtion is called after each iteration as
        callback( ||rk||_2 ), where rk is the current residual vector
    residuals : list
        residuals has the residual norm history,
        including the initial residual, appended to it
    reorth : boolean
        If True, then a check is made whether to reorthogonalize the Krylov 
        space each GMRES iteration

    Returns
    -------    
    (xNew, info)
    xNew : an updated guess to the solution of Ax = b
    info : halting status of gmres
        
            ==  =============================================
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.  This value
                is precisely the order of the Krylov space.
            <0  numerical breakdown, or illegal input
            ==  =============================================

    Notes
    -----
        - The LinearOperator class is in scipy.sparse.linalg.interface.
          Use this class if you prefer to define A or M as a mat-vec routine
          as opposed to explicitly constructing the matrix.  A.psolve(..) is
          still supported as a legacy.
        - For robustness, modified Gram-Schmidt is used to orthogonalize the Krylov Space
          Givens Rotations are used to provide the residual norm each iteration
    
    Examples
    --------
    >>> from pyamg.krylov import gmres
    >>> from pyamg.util.linalg import norm
    >>> import numpy 
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = numpy.ones((A.shape[0],))
    >>> (x,flag) = gmres(A,b, maxiter=2, tol=1e-8, orthog='mgs')
    >>> print norm(b - A*x)
    >>> 6.5428213057

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
       Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    .. [2] C. T. Kelley, http://www4.ncsu.edu/~ctk/matlab_roots.html
    '''
    # Convert inputs to linear system, with error checking  
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype)
    dimen = A.shape[0]

    # Choose type
    xtype = upcast(A.dtype, x.dtype, b.dtype, M.dtype)

    # Get fast access to underlying BLAS routines
    # dotc is the conjugate dot, dotu does no conjugation
    if xtype == scipy.complex64 or xtype == scipy.complex128 or xtype == scipy.complex192:
        [axpy,dotu,dotc,scal] = blas.get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], (x))
    else:    
        [axpy,dotu,dotc,scal] = blas.get_blas_funcs(['axpy', 'dot', 'dot',  'scal'], (x))

    # Make full use of direct access to blas by defining own norm
    def norm(z):
        return sqrt( real(dotc(z,z)) )


    # Should norm(r) be kept
    if residuals == []:
        keep_r = True
    else:
        keep_r = False

    # Set number of outer and inner iterations
    if restrt:
        if maxiter:
            max_outer = maxiter
        else:
            max_outer = 1
        if restrt > dimen:
            warn('Setting number of inner iterations (restrt) to maximimum allowed, which is A.shape[0] ')
            restrt = dimen
        max_inner = restrt
    else:
        max_outer = 1
        if maxiter > dimen:
            warn('Setting number of inner iterations (maxiter) to maximimum allowed, which is A.shape[0] ')
            maxiter = dimen
        elif maxiter == None:
            maxiter = min(dimen, 40)
        max_inner = maxiter

    # Scale tol by normb
    normb = norm(b) 
    if normb == 0:
        pass
    #    if callback != None:
    #        callback(0.0)
    #
    #    return (postprocess(zeros((dimen,)), dtype=xtype),0)
    else:
        tol = tol*normb

    # Is this a one dimensional matrix?
    if dimen == 1:
        entry = ravel(A*array([1.0], dtype=xtype))
        return (postprocess(b/entry), 0)

    # Prep for method
    r = b - ravel(A*x)
    normr = norm(r)
    if keep_r:
        residuals.append(normr)
    
    # Is initial guess sufficient?
    if normr <= tol:
        if callback != None:    
            callback(norm(r))
        
        return (postprocess(x), 0)

    #Apply preconditioner
    r = ravel(M*r)
    normr = norm(r)
    ## Check for nan, inf    
    #if isnan(r).any() or isinf(r).any():
    #    warn('inf or nan after application of preconditioner')
    #    return(postprocess(x), -1)
    
    # Use separate variable to track iterations.  If convergence fails, we cannot 
    # simply report niter = (outer-1)*max_outer + inner.  Numerical error could cause 
    # the inner loop to halt while the actual ||r|| > tol.
    niter = 0

    # Begin GMRES
    for outer in xrange(max_outer):
        
        # Preallocate for Given's Rotations, Hessenberg matrix and Krylov Space
        # Space required is O(dimen*max_inner).  
        # NOTE:  We are dealing with row-major matrices, so we traverse in a row-major fashion, 
        #        i.e., H and V's transpose is what we store.
        Q = []                                                 # Given's Rotations
        H = zeros( (max_inner+1, max_inner+1), dtype=xtype)    # Upper Hessenberg matrix, which is then 
                                                               #   converted to upper tri with Given's Rots 
        V = zeros( (max_inner+1, dimen), dtype=xtype)          # Krylov Space
        vs = []                                                # vs store the pointers to each column of V.
                                                               #   This saves a considerable amount of time.

        # v = r/normr
        V[0,:] = scal(1.0/normr, r)
        vs.append(V[0,:])

        # This is the RHS vector for the problem in the Krylov Space
        g = zeros((dimen,), dtype=xtype) 
        g[0] = normr

        for inner in xrange(max_inner):
            
            # New Search Direction
            v = V[inner+1,:]
            v[:] = ravel(M*(A*vs[-1]))
            vs.append(v)
            normv_old = norm(v)

            ## Check for nan, inf    
            #if isnan(V[inner+1, :]).any() or isinf(V[inner+1, :]).any():
            #    warn('inf or nan after application of preconditioner')
            #    return(postprocess(x), -1)

            #  Modified Gram Schmidt
            for k in xrange(inner+1):
                vk = vs[k]
                alpha = dotc(vk, v)
                H[inner, k] = alpha
                v[:] = axpy(vk, v, dimen, -alpha)

            normv = norm(v)
            H[inner, inner+1] = normv

            # Reorthogonalize
            if (reorth == True) and ( normv_old == normv_old + 0.001*normv):
                for k in xrange(inner+1):
                    vk = vs[k]
                    alpha = dotc(vk, v)
                    H[inner, k] = H[inner, k] + alpha
                    v[:] = axpy(vk, v, dimen, -alpha)

            # Check for breakdown
            if H[inner, inner+1] != 0.0:
                v[:] = scal(1.0/H[inner, inner+1], v)
            
            # Apply previous Given's rotations to H
            if inner > 0:
                apply_givens(Q, H[inner,:], inner)
                
            # Calculate and apply next complex-valued Given's Rotation
            # ==> Note that if max_inner = dimen, then this is unnecessary for the last inner 
            #     iteration, when inner = dimen-1.  
            if inner != dimen-1:
                if H[inner, inner+1] != 0:
                    h1 = H[inner, inner]; 
                    h2 = H[inner, inner+1];
                    h1_mag = abs(h1)
                    h2_mag = abs(h2)
                    if h1_mag < h2_mag:
                        mu = h1/h2
                        tau = conjugate(mu)/abs(mu)
                    else:    
                        mu = h2/h1
                        tau = mu/abs(mu)

                    denom = sqrt( h1_mag**2 + h2_mag**2 )               
                    c = h1_mag/denom
                    s = h2_mag*tau/denom; 
                    Qblock = array([[c, conjugate(s)], [-s, c]], dtype=xtype)
                    Q.append(Qblock)
                    
                    # Apply Given's Rotation to g, 
                    #   the RHS for the linear system in the Krylov Subspace.
                    g[inner:inner+2] = scipy.dot(Qblock, g[inner:inner+2])
                    
                    # Apply effect of Given's Rotation to H
                    H[inner, inner] = dotu(Qblock[0,:], H[inner, inner:inner+2]) 
                    H[inner, inner+1] = 0.0
                  
            # Don't update normr if last inner iteration, because 
            # normr is calculated directly after this loop ends.
            if inner < max_inner-1:
                normr = abs(g[inner+1])
                if normr < tol:
                    break
                
                # Allow user access to residual
                if callback != None:
                    callback( normr )
                if keep_r:
                    residuals.append(normr)
            
            niter += 1
            
        # end inner loop, back to outer loop

        # Find best update to x in Krylov Space, V.  Solve inner x inner system.
        y = scipy.linalg.solve(H[0:inner+1,0:inner+1].T, g[0:inner+1])
        update = ravel( scipy.mat(V[:inner+1,:]).T*y.reshape(-1,1) )
        x = x + update
        r = b - ravel(A*x)

        #Apply preconditioner
        r = ravel(M*r)
        normr = norm(r)
        ## Check for nan, inf    
        #if isnan(r).any() or isinf(r).any():
        #    warn('inf or nan after application of preconditioner')
        #    return(postprocess(x), -1)
        
        # Allow user access to residual
        if callback != None:
            callback( normr )
        if keep_r:
            residuals.append(normr)
        
        # Has GMRES stagnated?
        indices = (x != 0)
        if indices.any():
            change = max(abs( update[indices] / x[indices] ))
            if change < 1e-12:
                # No change, halt
                return (postprocess(x), -1)
    
        # test for convergence
        if normr < tol:
            return (postprocess(x),0)
            
    # end outer loop
    

    return (postprocess(x), niter)



if __name__ == '__main__':
    # from numpy import diag
    # A = random((4,4))
    # A = A*A.transpose() + diag([10,10,10,10])
    # b = random((4,1))
    # x0 = random((4,1))
    #%timeit -n 15 (x,flag) = gmres(A,b,x0,tol=1e-8,maxiter=100)
    
    from scipy.sparse import eye as speye
    from scipy import rand
    from pyamg.gallery import poisson
    from numpy.random import random
    from pyamg.util.linalg import norm
    A = poisson( (75,75), dtype=float,format='csr') 
    #A.data = A.data + 0.001j*rand(A.data.shape[0])
    b = random((A.shape[0],))
    x0 = random((A.shape[0],))

    import time
    from scipy.sparse.linalg.isolve import gmres as igmres

    print '\n\nTesting GMRES with %d x %d 2D Laplace Matrix'%(A.shape[0],A.shape[0])
    t1=time.time()
    (x,flag) = gmres_mgs(A,b,x0,tol=1e-8,maxiter=240)
    t2=time.time()
    print '%s took %0.3f ms' % ('gmres', (t2-t1)*1000.0)
    print 'norm = %g'%(norm(b - A*x))
    print 'info flag = %d'%(flag)

    t1=time.time()
    # DON"T Enforce a maxiter as scipy gmres can't handle it correctly
    (y,flag) = igmres(A,b,x0,tol=1e-8)
    t2=time.time()
    print '\n%s took %0.3f ms' % ('linalg gmres', (t2-t1)*1000.0)
    print 'norm = %g'%(norm(b - A*y))
    print 'info flag = %d'%(flag)

    
