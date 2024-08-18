import numpy as np
from   scipy.sparse import spdiags


FORWARD =1
BACKWARD= 2


def psi(t , w, q, mu, T, sigma, r): # analytical solution
   b = q * (r + (mu - r)** 2 / (2* (1 + q) * sigma ** 2))
   
   return np.exp(b * (t - T)) * np.power(w, q) / q

def f_y2(W, y, q, T, r): # terminal condition
        return (np.power(np.exp(W), q))/q * (y ** 2 + 1)

def inner_problem_solver(y, alph, Wmin, Wmax, N, A_max, J, q, T, dt, r, timesteps, eps_dyn_W, lam1, lam2, hsigsq, dWsq, sigmaxi, pi=0): #inner problem solver
    dW   = ( Wmax - Wmin) / N
    W    = np.linspace( Wmin, Wmax, N + 1 ) 
    Ps   = np.linspace( 0, A_max, J ) # we will do linear search in control space
    


    

    terminal_values = f_y2(W, y, q, T, r)   

        
    def alpha( W, p, eps,  dirn = FORWARD  ): # for upwinding scheme
        t1 = (W - W + hsigsq * (p**2)  / dWsq + eps ** 2)
        t2 = -1 * (( pi + ( r + p * sigmaxi ) - (hsigsq * (p**2)) ) + W - W) 
        if dirn == BACKWARD:
            return t1 - t2 / dW
        elif dirn == FORWARD:
            return t1
    
    def beta( W, p, eps, dirn = FORWARD ): # for upwinding scheme
        t1 = (W - W + hsigsq * (p**2)  / dWsq + eps ** 2)
        t2 = -1 * (( pi + ( r + p * sigmaxi ) - (hsigsq * (p**2)) ) + W - W) 
        if dirn == FORWARD:
            return t1 + t2 / dW
        elif dirn == BACKWARD:
            return t1    
        
    
    def makeDiagMat( alphas, betas ):
        d0, dl, d2 = -( alphas + betas ), np.roll( alphas, -1 ), np.roll( betas, 1 )
        d0[-1] = 0.
        dl [-2:] = 0.
        data = np.array( [ d0, dl, d2 ] )
        diags = np.array( [ 0, -1, 1 ] )
        return spdiags( data, diags, N + 1, N + 1 )
    
    def h(W , p, lam1, lam2):

        return lam1 * W + lam2 *  (1 - p) ** 2


    def find_optimal_ctrls( Vhat, t, eps_d, lam1, lam2 ): # we will find optimal controls using linear search in control space in every step in time for every point x 

        Fmin = np.tile( np.inf, Vhat.size )

        optdiffs = np.zeros_like( Vhat, dtype = np.int )
        optP    = np.zeros_like( Vhat )
        
        alphas  = np.zeros_like( Vhat ) 
        betas   = np.zeros_like( Vhat ) # the final
        curDiffs = np.zeros_like( Vhat, dtype = np.int )
        
        for p in Ps: 
            alphas[:] = -np.inf
            betas[:] = -np.inf
            curDiffs[:] = FORWARD
            
            for diff in [ FORWARD, BACKWARD ]:
                a = alpha( W, p,  eps_d, diff)
                b = beta( W, p, eps_d, diff )
                positive_coeff_indices = np.logical_and( a >= 0.0, b >= 0.0 ) == True
                positive_coeff_indices = np.logical_and( positive_coeff_indices, alphas==-np.inf )
                indices = np.where( positive_coeff_indices )

                alphas[ indices ] = a[ indices ]
                betas[ indices ] = b[ indices ]
                curDiffs[ indices ] = diff
                
            M = makeDiagMat( alphas, betas )
            F = M.dot( Vhat ) + h(W, p, lam1, lam2)
            indices = np.where( F < Fmin )
            
            Fmin[indices] = F[indices ]
            optP[indices] = p
            optdiffs[indices] = curDiffs[ indices ]
        return optP, optdiffs
    
    
    V = terminal_values
    time = len(timesteps) 
    alphas = np.zeros_like( V )
    betas = np.zeros_like( V )
    VV = np.zeros(len(V) * (len(timesteps) + 1))
    VV = VV.reshape((len(V), (len(timesteps) + 1)))
    VV[:, time] = V
    opt_c = np.zeros(len(V) * len(timesteps))
    opt_c = opt_c.reshape((len(V), len(timesteps)))
    
    for t in timesteps:
        
        time -= 1
        ctrls, optdiffs = find_optimal_ctrls( V, t, eps_dyn_W, lam1, lam2)
        for diff in [FORWARD, BACKWARD ]:  
                indices = np.where( optdiffs == diff )
                alphas[indices] = alpha( W[indices], ctrls[indices], diff )
                betas[indices] = beta( W[indices], ctrls[indices], diff )
            
        A   = dt * makeDiagMat( alphas, betas ) 
        Vnew = V + A.dot(V) +   h(W, ctrls, lam1, lam2) 
            
        V = Vnew
        
        opt_c[:, time] = ctrls    
        VV[:, time] = V   


    return W, V, ctrls, VV, opt_c

