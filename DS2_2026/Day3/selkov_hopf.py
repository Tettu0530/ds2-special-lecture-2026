import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
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


# Jacobian (symbolic)
xs, ys, as_, bs = sp.symbols("x y a b", real=True, positive=True)

f = -xs + as_ * ys + xs**2 * ys
g = bs - as_ * ys - (xs ** 2) * ys

solutions = sp.solve( [f, g]  , ( xs , ys ) )
fix_x , fix_y = solutions[0]

J_sym = sp.Matrix( [ [ f.diff( xs ) , f.diff( ys ) ] , [ g.diff( xs ) , g.diff( ys ) ] ] )


init = [ 1.0 , 1.0 ]
t_span = [ 0 , 100 ]
t_eval = np.linspace( t_span[0] , t_span[1] , 10000 )

b = 0.6

plt.figure( figsize=( 11 , 8 ) )

for i, a in enumerate( [ 0.14 , 0.06 ] ): # condition 1 / condition 2
    # Eigenvalues at the fixed point
    J_num = J_sym.subs( [ ( xs , fix_x ) , ( ys , fix_y ) , ( as_ , a ), ( bs , b ) ] )
    print( f'a = {a}' )
    print( J_num )
    print( J_num.eigenvals() )

    # Simulation
    param = [ a , b ]
    solution = solve_ivp( selkov_model, t_span, init, t_eval=t_eval, method='RK45', args=param )

    ADP = solution.y[0]
    F6P = solution.y[1]

    # Time series
    plt.subplot( 2 , 2 , i + 1 )
    plt.title( f'a = {a}: time series' )
    plt.plot( solution.t , ADP )
    plt.plot( solution.t , F6P )
    plt.xlabel( 't' )
    plt.ylabel( 'concentration' )

    # Phase plane
    x = np.linspace( 0, 3, 100 )
    x_null = x / ( a + x ** 2 ) # The nullcline for dxdt
    y_null = b / ( a + x ** 2) # The nullcline for dydt

    plt.subplot( 2 , 2 , i + 3 )
    plt.title( f'a = {a}: phase plane' )
    plt.plot( x, x_null, 'r-')
    plt.plot( x, y_null, 'b-')
    plt.plot( ADP, F6P, 'k-')
    plt.ylim( 0 , 4 )
    plt.xlabel( 'ADP (x)' )
    plt.ylabel( 'F6P (y)' )

plt.tight_layout()
plt.show()
