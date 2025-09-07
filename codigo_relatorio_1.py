import matplotlib.pyplot as plt 
import control as ct 
import numpy as np 

#Leonardo Torres sistema
t = np.linspace(0.0, 2001.0, 1000) 
uf = np.array([12.0 for i in range(len(t))]) 

pmt = {'roh':1.0, 'g':9.81, 'a':-0.1, 'R0':1.0, 'R1':5.0, 'H':4.0, 
       'La':0.1, 'ra':1.0, 'k1':1.19, 'Vmax':12.0, 'k5':1.0, 'k6': 4.0, 
       'J':30.0, 'b':1/np.pi, 'k2':1.19, 'k3': 1.0} 

def states_dx(t, x, u, params):
    u = np.atleast_1d(u)[0] 
    roh, g, a, R0, R1, H = map(params.get, ['roh', 'g', 'a', 'R0', 'R1', 'H'])
    La, ra, k1, Vmax, k5 = map(params.get, ['La', 'ra', 'k1', 'Vmax', 'k5'])
    J, b, k2, k3 = map(params.get, ['J', 'b', 'k2', 'k3'])
    
    h = float(np.clip(x[2], 0.0, 4.0))      # limitação do valor da medida da altura do tanque, que deve ficar entre 0.0 e 4.0
    theta = float(np.clip(x[2], 0.0, 2*np.pi))  # limitação do Ângulo de posição da válvula, entre 0 e 2pi.
    r_h = float(R0 + (R1 - R0) * h / H)
    va = Vmax * (1 - 2 / (1 + np.exp(k5 * (u - 12))))
    c_v = float(-a * np.pi / (np.sqrt(roh * g)))
    
    k4 = float((1 / b) * (abs(a * np.sqrt(H)) * 1.2) / (np.exp(k3 * 1.835 * np.pi)))
    qi = float(k4 * (np.exp(k3 * theta) - 1.0))

    dx0 = ((-ra * x[0] - x[1] * k1 + va) / La)
    dx1 = float((k2 * x[0] - b * x[1]) / J)
    dx2 = float(x[1])
    dx3 = float((-(c_v * np.sqrt(roh * g * h)) + qi) / max(np.pi * r_h**2, 1e-6))

    return np.array([dx0, dx1, dx2, dx3], dtype=float)


def output_y(t, x, u, params): 
    H, k6 = map(params.get, ['H', 'k6']) 
    k7 = 16.0/H 
    y = k6 + k7*x[3] 
    return y 

sys = ct.NonlinearIOSystem(
    updfcn=states_dx, 
    outfcn=output_y, 
    inputs=1, 
    outputs=1,
    states=4,
    name='tanque conico',
    params=pmt)
 
ra = pmt.get('ra') 
 
x0_guess = np.array([0.0, 0.0, 1.1, 1.0])  
xeq, ueq = ct.find_operating_point(sys, x0_guess, np.array([12.0]), params=pmt)

x_0 = np.array([0.0, 0.0, 5.58625704, 1.991])       
x_02 =  np.array([0.0, 0.0, 0.09, 1.0]) 
resp = ct.input_output_response(sys, t, uf, x_0) 
t = resp.time 
y = resp.outputs 
s = resp.states 


def sys_linearized(params, xeq, ueq):
    roh, g, a, R0, R1, H = map(params.get, ['roh','g','a','R0','R1','H'])
    La, ra, k1, Vmax, k5 = map(params.get, ['La','ra','k1','Vmax','k5'])
    J, b, k2, k3 = map(params.get, ['J','b','k2','k3'])
    k4 = (1/b)*(abs(a*np.sqrt(H))*1.2)/np.exp(k3*1.835*np.pi)
    k7 = 16.0/H

    h_eq = xeq[2]


    # Aproximação da derivada parcial
    A32 = (4*b*k3*(k4 - a/b*np.sqrt(h_eq)))/((R1-R0)**2)
    A33 = ((4*a*np.sqrt(h_eq))/((R1-R0)**3)) * ((5*R1-3*R0)/(4*H)-1)

    A = np.array([[-ra/La, -k1/La, 0, 0],
                  [k2/J, -b/J, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, A32, A33]])

    B0 = Vmax*k5/(2*La)
    B = np.array([B0, 0, 0, 0]).reshape(4,1)
    C = np.array([0, 0, 0, k7]).reshape(1,4)
    D = np.array([[0]])

    return ct.ss(A,B,C,D)

sys_lin = sys_linearized(pmt, x_0, ueq)

# Simular resposta linear em desvios
delta_u = uf - ueq
delta_x0 = np.zeros_like(xeq)  # assumindo início no ponto de equilíbrio
t_lin, y_lin = ct.forced_response(sys_lin, t, delta_u, X0=x_02)
y_lin += output_y(0, xeq, ueq, pmt)  # adiciona ponto de equilíbrio à saída
    

plt.plot(t, y, 'b', label='resposta não linear') 
plt.plot(t, uf, 'k--', label='referência') 
plt.plot(t, y_lin, 'r', label='resposta linear') 
plt.ylim(0.0, 20.0)
plt.legend() 
plt.xlabel('t') 
plt.ylabel('y(t)') 
plt.grid() 
plt.show() 

















