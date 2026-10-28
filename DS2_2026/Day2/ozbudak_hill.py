import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def ozbudak_model( t , init , *args ):
    x = init[ 0 ]
    y = init[ 1 ]

    [ alpha , beta , rho , x_out, n ] = args

    dxdt = -x + alpha * x_out + beta * y
    dydt =  x**n / ( rho + x**n ) - y
    
    return [ dxdt , dydt ] 

def ozbudak_dose_response( x_out , y_init , marker, n ):
    init = [ 0 , y_init ]
    t_span = [ 0 , 60 ]

    alpha = 0.1
    beta = 10
    rho = 25
    
    solution = solve_ivp( ozbudak_model, t_span, init , method='RK45', args=[ alpha , beta , rho , x_out, n ] )
    
    LacY = solution.y[1]
    y_end = LacY[-1]
    
    plt.plot( x_out , y_end , marker , fillstyle='none') 

    return y_end
    
##
## Main
##

for n in range(1, 6):
    y_init = 0

    plt.subplot(2, 3, n)
    plt.title( f'Hill coefficient: {n}' )
    plt.xlabel( 'Xout (TMGout)' )
    plt.ylabel( 'y (LacY-GFP end)' )

    for x_out in range(11):
        y_init = ozbudak_dose_response( x_out , y_init , 'ko', n )

    for x_out in reversed( range(11) ):
        y_init = ozbudak_dose_response( x_out, y_init, 'r+', n )

plt.tight_layout()
plt.show()
