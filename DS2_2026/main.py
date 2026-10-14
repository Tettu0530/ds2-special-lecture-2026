import matplotlib.pyplot as plt

glu = 1.0
t = 0.0
delta_t = 0.01
endtime = 5.0

k = 0.2

glu_list = [ glu ]
t_list = [ t ]

while t < endtime - delta_t / 2 :
    dgludt = - k * glu
    delta_glu = dgludt * delta_t
    glu = glu + delta_glu
    t = t + delta_t

    glu_list.append( glu )
    t_list.append( t )

plt.plot( t_list , glu_list )
plt.xlabel( "t(sec)" )
plt.ylabel( "[Glu](mM)" )
plt.show()
