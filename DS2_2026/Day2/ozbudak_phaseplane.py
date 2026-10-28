import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def ozbudak_model( t , init , *param ):
    x = init[ 0 ]
    y = init[ 1 ]

    [ alpha , beta , rho , x_out ] = param
    
    dxdt = -x + alpha * x_out + beta * y
    dydt = (x ** 2) / (rho + (x ** 2)) - y
    
    return [ dxdt , dydt ] 


init = [ 3 , 0 ]
t_span = [ 0 , 30 ]

alpha = 0.1
beta = 10
rho = 25

for i, x_out in enumerate( [ 3 , 10 ] ): # without / with hysteresis
    param = [ alpha , beta , rho , x_out ]

    solution = solve_ivp( ozbudak_model, t_span, init , method='RK45', args=param )

    TMG = solution.y[0]
    LacY = solution.y[1]

    plt.subplot( 1 , 2 , i + 1 )
    plt.title( f'x_out = {x_out}' )
    plt.xlabel( 'x (TMG)' )
    plt.ylabel( 'y (LacY-GFP)' )

    plt.plot(TMG, LacY, 'k-')

    x = np.linspace( 0, 10, 100 )

    x_null = ( x - alpha * x_out ) / beta # The nullcline for dxdt=0
    y_null = x ** 2 / (rho + x ** 2) # The nullcline for dydt=0

    plt.plot( x , x_null , 'r-')
    plt.plot( x , y_null , 'b-')

plt.tight_layout()
plt.show()
