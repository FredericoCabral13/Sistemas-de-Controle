import numpy as np
import matplotlib.pyplot as plt
import control as ct

# %% QUESTÃO 1

# planta G = 1/(s+1)^6:
G = ct.tf([1.0],[1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
t = np.linspace(0.0, 50.0, 100)
t, yp = ct.step_response(G, t)

K = 1
# tau = 2.0
# L = 4.0

t632 = 6.504 
t284 = 4.4187

tau = 1.5*(t632 - t284)
L = 1.5*(t284 - t632/3)

G_1o = ct.tf([K], [tau, 1])

num_pade, den_pade = ct.pade(L, 1)
Pade = ct.tf(num_pade, den_pade)
G_pade = G_1o * Pade
Gcl = ct.feedback(G_pade, 1)

t, y_aprox = ct.step_response(G_pade, t)
t, y_aprox2 = ct.step_response(Gcl, t)

plt.plot(t, yp, 'b', label='Sistema real')
plt.plot(t, y_aprox, 'r', label='Modelo aproximado')
plt.title('Resposta em malha aberta')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()


#Controlador projetado

Kc = 0.3
z2 = 0.4*(3.13+Kc)/Kc

Kp = 1.97
Ki = 0.53
Kd = 1.0
Tf = 0.02  # Filtro ajustado

# 3. Construção do PID com filtro (C = Kc*(Kp + Ki/s + (Kd * s) / (Tf * s + 1)))
C = Kc*(ct.tf([Kp],1) + ct.tf([Ki],[1,0]) + ct.tf([Kd],[Tf,1]))



Gcl_2 = ct.feedback(C*G, 1)
t, y = ct.step_response(Gcl_2, t)

# plt.plot(t, y_aprox2, 'b', label='Resposta aproximada fechada')
plt.plot(t, y, 'b', label='Resposta controlada')
plt.axhline(1.0, color='r', linestyle='--', label='Referência')
# plt.title('Resposta em malha aberta')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

#Aplicando preditor de Smith na aproximação

Ceq = ct.feedback(C, G_1o - G_pade)
Gcl_3 = ct.feedback(Ceq*G_pade, 1)
t,y_smith_1 = ct.step_response(Gcl_3, t)

plt.plot(t, y_aprox2, 'b', label='Resposta do modelo para MF')
plt.plot(t, y_smith_1, 'r', label='Resposta com Preditor de Smith')
plt.axhline(1.0, color='g', linestyle='--', label='Referência')
plt.title('P. Smith p/ aproximação de 1ª ordem')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

#Aplicando preditor de Smith no sistema real

# L_2 = 0.5 #visto no gráfico
# num_pade_2, den_pade_2 = ct.pade(L_2, 6)
# Pade_2 = ct.tf(num_pade_2, den_pade_2)
# G_pade_2 = G * Pade_2

# G_sem_atr = G - G_pade_2

Ceq_2 = ct.feedback(C, G_1o - G_pade)
Gcl_4 = ct.feedback(Ceq_2*G, 1)
t,y_smith_2 = ct.step_response(Gcl_4, t)


plt.plot(t, y, 'b', label='Resposta real em MF')
plt.plot(t, y_smith_2, 'r', label='Resposta com Preditor de Smith')
plt.axhline(1.0, color='g', linestyle='--', label='Referência')
plt.title('P. Smith p/ sistema real')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()


# %% QUESTÃO 2

import numpy as np
import matplotlib.pyplot as plt
import control as ct

t = np.linspace(0.0, 10.0, 100)    # Array de tempo               
ref = np.array([1.0 for i in range(len(t))])  # Array de referência

G2 = ct.tf([3.0], [1.0, 2.0, 5.0], name='G2', inputs='u', outputs='y')

t_out, y_op = ct.forced_response(G2, T=t, U=ref)

plt.plot(t, y_op, 'r', label='saida do sistema em malha aberta')
plt.xlabel('t (seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

up = 5.0
tau = 1.0
tau_mfd = tau
zt = -np.log(up/100)/(np.sqrt(np.pi**2 + (np.log(up/100))**2))
wn = 4/(4*tau_mfd*zt)

kp = (wn**2)*2.0/3.0
ki = (wn**2)*5.0/3.0
kd = (wn**2)/3.0

P = ct.tf([kp], [1.0], name='P', inputs='u', outputs='y')
I = ct.tf([1.0], [1.0, 0.0], name='I', inputs='u', outputs='y')
KI = ct.tf([ki], [1.0], name='KI', inputs='u', outputs='y')
D = ct.tf([kd, 0.0], [0.0001, 1.0], name='D', inputs='u', outputs='y')
H = ct.tf([1], [1], name="H", inputs="u", outputs="y")  
sum_err = ct.summing_junction(inputs=["r", "-uH"], output="y", name="sum_err")
sum_control = ct.summing_junction(inputs=["p", "i", "d"], output="y", name="sum_control")
sum_anti_w = ct.summing_junction(inputs=["-w", "u"], output="y", name="sum_anti_w")
sum_intg = ct.summing_junction(inputs=["ic", "ies"], output="y", name="sum_intg")
Ies = ct.tf([1/(np.sqrt((kp/ki)*(kp*kd)))], [1.0], name='Ies', inputs='u', outputs='y')
f = ct.tf([1.0], [1.0, 2*zt*wn], name='f', inputs="u", outputs="y")
u_max = 1.77
u_sat = 0.7 * u_max
PID = P + KI + D
C = PID*f

def out_fcn(t, x, u, params=None):
    y = np.array([0.0])
    if u[0] > u_sat:
        y[0] = u_sat
    elif u[0] < -u_sat:
        y[0] = -u_sat
    else:
        y[0] = u[0]
    return y

def updfcn(t, x, u, params=None):
    return []

atuador = ct.NonlinearIOSystem(
    updfcn=updfcn,
    outfcn=out_fcn,
    inputs=1,
    outputs=1,
    states=0,
    name="atuador"
)

malha_f = ct.interconnect(
    syslist=[G2, P, I, D, KI, H, sum_err, sum_control, f, sum_anti_w, sum_intg, Ies, atuador],
    connections=[
        ['sum_err.uH', 'H.y'], 
        ['P.u', 'sum_err.y'],       
        ['KI.u', 'sum_err.y'],      
        ['D.u', 'sum_err.y'],      
        ['sum_control.p', 'P.y'],
        ['sum_control.i', 'I.y'],
        ['sum_control.d', 'D.y'],
        ['f.u', 'sum_control.y'],
        ['G2.u', 'atuador.y'],
        ['H.u', 'G2.y'],
        ['sum_anti_w.w', 'f.y'],
        ['sum_anti_w.u', 'atuador.y'], 
        ['atuador.u', 'f.y'],
        ['Ies.u', 'sum_anti_w.y'],
        ['sum_intg.ic', 'KI.y'],
        ['sum_intg.ies', 'Ies.y'],
        ['I.u', 'sum_intg.y'],
    ],
    inplist=["r"],
    outlist=["G2.y", "atuador.y", "f.y"]   
)

t_out, Y = ct.forced_response(malha_f, T=t, U=ref)
y = Y[0, :]
u_w = Y[1, :]
u = Y[2, :]


plt.plot(t, y, 'b', label='saida do sistema em malha fechada')
plt.xlabel('t(seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t, u, 'r', label='sinal de controle para a malha fechada')
plt.xlabel('t(seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t, u_w, 'r', label='sinal de controle para a malha fechada com o anti-windup')
plt.xlabel('t(seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

Gcl = ct.feedback((G2*C), H)
t_out, Y2 = ct.forced_response(Gcl, T=t, U=ref)

plt.plot(t, Y2, 'r', label='saida do sistema em malha aberta')
plt.xlabel('t(seg)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()


G_inv = 1/G2
print(f'teste inv planta= {G_inv}')








# %% QUESTÂO 3

import numpy as np
import matplotlib.pyplot as plt
import control as ct

# ==========================================
# 1. DEFINIÇÃO DA PLANTA E PARÂMETROS
# ==========================================
# Planta assumida (do exercício anterior). Substitua se tiver outra.
# G(s) = 3 / (s^2 + 2s + 5)
num = [3]
den = [1, 2, 5]
sys_cont = ct.tf(num, den)

# Conversão para Espaço de Estados (para simulação passo a passo)
dt = 0.01
T_sim = 20.0
t = np.arange(0, T_sim, dt)
N = len(t)
sys_d = ct.c2d(ct.ss(sys_cont), dt, method='zoh')
A, B, C, D = sys_d.A, sys_d.B, sys_d.C, sys_d.D


dead_zone_limit = 0.2  # 20% do sinal unitário

def apply_dead_zone(u_in, limit):
    if abs(u_in) < limit:
        return 0.0
    elif u_in > limit:
        return u_in - limit 
    else: # u_in < -limit
        return u_in + limit


# Controlador P (Tentativa e erro ou lugar das raízes)
Kp_only = 2.0 

# Controlador PI (Sintonia moderada para evitar oscilação excessiva na não-linearidade)
Kp_pi = 2.0
Ki_pi = 1.5


ref = 0.25 # Referência baixa, próxima da zona de perigo

def simular(controlador_tipo):
    y = np.zeros(N)
    u_calc = np.zeros(N) # Sinal calculado pelo PID
    u_eff = np.zeros(N)  # Sinal efetivo que entra na planta (após Dead Zone)
    x = np.zeros((2, 1))
    
    integral = 0.0
    erro_ant = 0.0
    
    for k in range(N-1):
        erro = ref - y[k]
        
        # Lógica de Controle
        if controlador_tipo == 'P':
            u = Kp_only * erro
        else: # PI
            integral += erro * dt
            u = Kp_pi * erro + Ki_pi * integral
            
        u_calc[k] = u
        
        # Aplicação da ZONA MORTA
        u_in_plant = apply_dead_zone(u, dead_zone_limit)
        u_eff[k] = u_in_plant
        
        # Atualiza Planta
        x_next = A @ x + B * u_in_plant
        y_next = C @ x + D * u_in_plant
        x = x_next
        y[k+1] = y_next.item()
        
    return y, u_calc, u_eff

# Executa Simulações
y_p, u_calc_p, u_eff_p = simular('P')
y_pi, u_calc_pi, u_eff_pi = simular('PI')


plt.figure(figsize=(12, 10))

# Saída y(t)
plt.subplot(3, 1, 1)
plt.title(f'Comparativo P vs PI com Zona Morta de {dead_zone_limit*100}%')
plt.plot(t, y_p, 'r--', label='Controlador P')
plt.plot(t, y_pi, 'b-', label='Controlador PI')
plt.plot(t, np.ones(N)*ref, 'k:', label='Referência')
plt.ylabel('Saída (Posição/Velocidade)')
plt.legend()
plt.grid(True)

# Sinal Calculado pelo Controlador
plt.subplot(3, 1, 2)
plt.plot(t, u_calc_p, 'r--', label='Saída do P (Calculada)')
plt.plot(t, u_calc_pi, 'b-', label='Saída do PI (Calculada)')
plt.hlines([dead_zone_limit, -dead_zone_limit], 0, T_sim, 'k', linestyles='dotted', label='Limites Zona Morta')
plt.ylabel('Sinal de Controle (u)')
plt.legend()
plt.grid(True)

# Sinal Efetivo (O que a planta realmente vê)
plt.subplot(3, 1, 3)
plt.plot(t, u_eff_p, 'r--', label='Entrada Efetiva P')
plt.plot(t, u_eff_pi, 'b-', label='Entrada Efetiva PI')
plt.ylabel('Sinal Pós-Zona Morta')
plt.xlabel('Tempo (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()