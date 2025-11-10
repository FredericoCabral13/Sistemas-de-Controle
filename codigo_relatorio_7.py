import numpy as np
import matplotlib.pyplot as plt
import control as ct

# Modelo em malha aberta:
k = 1872.0
tau = 4829.0
G = ct.tf([k], [tau, 1.0])
t = np.linspace(0.0, 75000.0, 75000)
t, yop = ct.step_response(G, t)

plt.plot(t, yop, 'b', label='resposta em malha aberta')
plt.title('resposta em malha aberta para o melhor modelo obtido para a FanPlate')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

#Projetando um controlador via síntese direta:

kmd = 1.0
tau_md = tau/10.0   # Quanto mais eu diminuir a constante de tempo do sistema (fração do cosntante de tempo da planta), mais rápido o sistema em malha fechada atingirá a referência
C = ct.tf([tau/(k*tau_md), 1.0/(k*tau_md)], [1.0, 0.0])

# Fechando a malha com o controlador:

Gcl = ct.feedback((C*G), 1)     # Realimentação considerando um sensor ideal

# Gerando a resposta para a malha fechada com o controlador:
t, ycl = ct.step_response(Gcl, t)
plt.plot(t, ycl, 'b', label='resposta em malha fechada com o controlador projetado (PI)')
plt.title('resposta em malha fechada')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# Comparando com a  dinâmica em malha aberta para diferentes valores de constante de tempo (2tau, 5tau e 10tau)

tau2 = 2*tau
tau5 = 5*tau
tau10 = 10*tau

C2 = ct.tf([tau/(k*tau2), 1.0/(k*tau2)], [1.0, 0.0])
C5 = ct.tf([tau/(k*tau5), 1.0/(k*tau5)], [1.0, 0.0])
C10 = ct.tf([tau/(k*tau10), 1.0/(k*tau10)], [1.0, 0.0])

Gcl2 = ct.feedback((C2*G), 1) 
Gcl5 = ct.feedback((C5*G), 1) 
Gcl10 = ct.feedback((C10*G), 1) 

t, ycl2 = ct.step_response(Gcl2, t)
t, ycl5 = ct.step_response(Gcl5, t)
t, ycl10 = ct.step_response(Gcl10, t)

plt.plot(t, ycl2, 'b', label='resposta em malha fechada para τ_c = 2*τ')
plt.plot(t, ycl5, 'r', label='resposta em malha fechada para τ_c = 5*τ')
plt.plot(t, ycl10, 'g', label='resposta em malha fechada para τ_c = 10*τ')
plt.title('Comparação das respostas considerando diferentes tau_c')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

r = np.ones_like(t)

# Avaliação quantitativa
def rmse(y_r, y_m):
    return np.sqrt(1/(len(t)) * np.sum((y_r - y_m) * (y_r - y_m)))

print(f'\nRMSE para a malha aberta em relação a referência : {rmse(yop, r)}\n')
print(f'RMSE para a malha fechada com 2 tau em relação a referência : {rmse(ycl2, r)}\n')
print(f'RMSE para a malha fechada com 5 tau em relação a referência : {rmse(ycl5, r)}\n')
print(f'RMSE para a malha fechada com 10 tau em relação a referência : {rmse(ycl10, r)}\n')

# Avaliação qualitativa: 

# Quanto menor for a constante de tempo da malha fechada, mais
#rápido ele irá se acomodar no valor da referência.

# Como a resposta em malha aberta se estabiliza em um valor diferente da referência, o
# valor do RMSE tende a ser mais maior para ele, pois existe um erro até quando esse 
# sistema estabiliza ( erro em regime permanente não nulo).

# Repetindo o exercício para uma malha fechada desejada de segunda ordem:

# considerando uma constante de tempo 10 vezes menor, temos:

tau_md2 = tau/10.0
UP = 5.0

zt = -np.log(UP/100.0)/(np.sqrt((np.pi**2) + (np.log(UP/100.0))**2))
wn = 1.0/(zt*tau_md2)

PI = ct.tf([tau/k, 1.0/k], [1.0, 0.0])
f1 = ct.tf([wn**2], [1.0, 2*zt*wn])
C_2o = PI*f1

Gcl_2o = ct.feedback((C_2o*G), 1)     # Realimentação considerando um sensor ideal

# Gerando a resposta para a malha fechada com o controlador:
t, ycl_2o = ct.step_response(Gcl_2o, t)
plt.plot(t, ycl_2o, 'b', label='resposta em malha fechada com o controlador projetado (PI) em série com um filtro')
plt.title('resposta em malha fechada desejada de segunda ordem')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# Comparando com a  dinâmica em malha aberta para diferentes valores de constante de tempo (2tau, 5tau e 10tau)


wn2 = 1.0/(zt*tau2)
wn5 = 1.0/(zt*tau5)
wn10 = 1.0/(zt*tau10)

f1_2 = ct.tf([wn2**2], [1.0, 2*zt*wn2])
f1_5 = ct.tf([wn5**2], [1.0, 2*zt*wn5])
f1_10 = ct.tf([wn10**2], [1.0, 2*zt*wn10])

C2_2o = PI*f1_2
C5_2o = PI*f1_5
C10_2o = PI*f1_10

Gcl2_2o = ct.feedback((C2_2o*G), 1) 
Gcl5_2o = ct.feedback((C5_2o*G), 1) 
Gcl10_2o = ct.feedback((C10_2o*G), 1) 

t, ycl2_2o = ct.step_response(Gcl2_2o, t)
t, ycl5_2o = ct.step_response(Gcl5_2o, t)
t, ycl10_2o = ct.step_response(Gcl10_2o, t)

plt.plot(t, ycl2_2o, 'b', label='resposta em malha fechada para τ_c = 2*τ')
plt.plot(t, ycl5_2o, 'r', label='resposta em malha fechada para τ_c = 5*τ')
plt.plot(t, ycl10_2o, 'g', label='resposta em malha fechada para τ_c = 10*τ')
plt.title('Comparação das respostas considerando diferentes tau_c')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

r = np.ones_like(t)

# Avaliação quantitativa

print(f'\nRMSE para a malha aberta em relação a referência : {rmse(yop, r)}\n')
print(f'RMSE para a malha fechada com 2 tau em relação a referência : {rmse(ycl2_2o, r)}\n')
print(f'RMSE para a malha fechada com 5 tau em relação a referência : {rmse(ycl5_2o, r)}\n')
print(f'RMSE para a malha fechada com 10 tau em relação a referência : {rmse(ycl10_2o, r)}\n')
