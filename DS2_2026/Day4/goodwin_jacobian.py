import sympy as sp

# 変数と定数（記号）
v, u = sp.symbols("v u", real=True)
alpha, beta, gamma, rho, sigma = sp.symbols("alpha beta gamma rho sigma", real=True, positive=True)

param_val = [ ( alpha , 0.02 ) , ( beta , 0.01 ) , ( gamma , 0.01 ) , ( rho , 0.5 ) , ( sigma , 2.0 ) ]

f = ( ( 1 / sigma - ( alpha + beta ) ) - u / sigma ) * v  # dvdt
g = ( - ( alpha + gamma ) + rho * v ) * u                 # dudt

# ヌルクラインと固定点
solutions = sp.solve( [ f , g ] , ( v , u ) )
print( solutions )

# v = 0 や u = 0 を含む解は経済的に意味がないので除く
fix_v , fix_u = [ sp.simplify( c ) for c in [ s for s in solutions if s[0] != 0 and s[1] != 0 ][0] ]
print( fix_v , fix_u )
print( fix_v.subs( param_val ) , fix_u.subs( param_val ) )

# ヤコビ行列
J_sym = sp.Matrix( [ [ f.diff( v ) , f.diff( u ) ] , [ g.diff( v ) , g.diff( u ) ] ] )
J_fix = sp.simplify( J_sym.subs( [ ( v , fix_v ) , ( u , fix_u ) ] ) )  # 固定点での J（記号のまま）
J_num = J_fix.subs( param_val )

print( J_sym )
print( J_fix )
print( J_num )
print( J_num.eigenvals() )
