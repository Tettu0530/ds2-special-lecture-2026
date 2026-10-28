import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def selkov_model( t , init , *param ):
    x = init[ 0 ]
    y = init[ 1 ]

    [ a , b ] = param


    v1 = b
    v2 = a * y + (x ** 2) * y
    v3 = x

    dxdt = v2 - v3
    dydt = v1 - v2

    return [ dxdt , dydt ]


init = [ 1.0 , 1.0 ]
t_span = [ 0 , 100 ]
t_eval = np.linspace( t_span[0] , t_span[1] , 10000 )

a = 0.06
b = 0.6

param = [ a , b ]

# Trajectory
solution = solve_ivp( selkov_model, t_span, init, t_eval=t_eval, method='RK45', args=param )

ADP = solution.y[0]
F6P = solution.y[1]

# Nullclines
x = np.linspace( 0, 3, 100 )
x_null = x / ( a + x ** 2 ) # The nullcline for dxdt
y_null = b / ( a + x ** 2) # The nullcline for dydt

# Vector field (grid names xg, yg to avoid overwriting x)
N = 10
xg = np.linspace( 0, 3, N )
yg = np.linspace( 0, 10, N )

[X, Y] = np.meshgrid( xg, yg )

dXdt = ( a * Y + (X ** 2) * Y ) - X # v2 - v3
dYdt = b - ( a * Y + (X ** 2) * Y ) # v1 - v2

for i, method in enumerate( [ 'quiver' , 'streamplot' ] ):
    plt.subplot( 1 , 2 , i + 1 )
    plt.title( method )

    if method == 'quiver':
        plt.quiver( X, Y, dXdt, dYdt )
    else:
        plt.streamplot( X, Y, dXdt, dYdt )

    plt.plot( x, x_null, 'r-')
    plt.plot( x, y_null, 'b-')
    plt.plot( ADP, F6P, 'k-')

    plt.xlabel( 'ADP (x)' )
    plt.ylabel( 'F6P (y)' )

plt.tight_layout()
plt.show()
