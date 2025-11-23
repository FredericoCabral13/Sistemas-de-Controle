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

num = [3]
den = [1, 2, 5]
sys_cont = ct.tf(num, den)

# Conversão para Espaço de Estados (necessário para simulação matricial)
sys_ss = ct.ss(sys_cont)

# Parâmetros de Tempo
dt = 0.01           # Passo de tempo (s)
T_sim = 12.0        # Tempo total de simulação
t = np.arange(0, T_sim, dt)
N = len(t)

# Discretização da Planta (Zero-Order Hold)
sys_d = ct.c2d(sys_ss, dt, method='zoh')
A, B, C, D = sys_d.A, sys_d.B, sys_d.C, sys_d.D


tau_c = 1.0 

# Cálculo dos Ganhos PID
Kp = 2 / (3 * tau_c)
Ki = 5 / (3 * tau_c)
Kd = 1 / (3 * tau_c)

# Regra de projeto: Kb = 1/tau_c
Kb = 1.0 / tau_c

print(f"Ganhos Calculados:\nKp = {Kp:.4f}\nKi = {Ki:.4f}\nKd = {Kd:.4f}\nKb = {Kb:.4f}")


def simular_sistema(tipo_simulacao, u_max_val=None):
    
    y_out = np.zeros(N)
    u_out = np.zeros(N)     # Sinal real aplicado
    v_out = np.zeros(N)     # Sinal calculado (interno do PID)
    
    x = np.zeros((2, 1))    # Estado inicial da planta
    integral = 0.0
    erro_ant = 0.0
    
    ref = 1.0 # Degrau unitário
    
    for k in range(N-1):
        # 1. Cálculo do Erro
        erro = ref - y_out[k]
        
        # 2. Termos do PID
        P = Kp * erro
        deriv = (erro - erro_ant) / dt
        D_term = Kd * deriv
        
        # O termo integral (I) é acumulado abaixo dependendo do método
        
        # 3. Sinal de Controle Calculado (v)
        v = P + integral + D_term
        v_out[k] = v
        
        # 4. Lógica de Saturação e Anti-Windup
        if tipo_simulacao == 'linear':
            u = v
            # Integração padrão
            integral += Ki * erro * dt
            
        else: # Casos com saturação
            # Aplica Saturação
            if v > u_max_val:
                u = u_max_val
            elif v < -u_max_val:
                u = -u_max_val
            else:
                u = v
            
            # Aplica Anti-Windup
            if tipo_simulacao == 'back_calc':
                # Fórmula do Back Calculation: I_dot = Ki*e + Kb*(u - v)
                erro_sat = u - v
                integral += (Ki * erro + Kb * erro_sat) * dt
            
            elif tipo_simulacao == 'windup':
                # Erro clássico: continua integrando mesmo saturado
                integral += Ki * erro * dt
        
        u_out[k] = u
        erro_ant = erro
        
        # 5. Atualização da Planta (Espaço de Estados)
        x_next = A @ x + B * u
        y_next = C @ x + D * u
        
        x = x_next
        y_out[k+1] = y_next.item()
        
    return y_out, u_out, v_out


# PASSO A: Simulação Linear para descobrir o Pico
_, _, v_linear = simular_sistema('linear')
pico_livre = np.max(v_linear)
u_ss_teorico = 5/3 # 1.66...

print(f"\nPico Máximo (Resposta Livre): {pico_livre:.4f}")
print(f"Esforço de Regime (Steady State): {u_ss_teorico:.4f}")

# PASSO B: Definir Limite de Saturação (70%)
u_limite = 0.70 * pico_livre
print(f"Limite de Saturação Definido (70%): {u_limite:.4f}")

# Validação de segurança
if u_limite < u_ss_teorico:
    print("ALERTA CRÍTICO: O limite de 70% é menor que o valor necessário para regime permanente!")
    print("O sistema nunca chegará na referência. Aumentando tau_c seria necessário.")
else:
    print("Condição de projeto OK: O limite permite chegar ao regime permanente.")

# PASSO C: Simulações Comparativas
# 1. Com Windup (Problemático)
y_bad, u_bad, v_bad = simular_sistema('windup', u_limite)

# 2. Com Back Calculation (Correto)
y_good, u_good, v_good = simular_sistema('back_calc', u_limite)


plt.figure(figsize=(12, 10))

# Gráfico 1: Saída y(t)
plt.subplot(3, 1, 1)
plt.title(f'Comparação de Desempenho (Saturação em {u_limite:.2f})')
plt.plot(t, y_bad, 'r--', label='Sem Anti-Windup (Windup)')
plt.plot(t, y_good, 'b-', linewidth=2, label='Com Back Calculation')
plt.plot(t, np.ones(N), 'k:', label='Referência')
plt.ylabel('Saída y(t)')
plt.legend(loc='lower right')
plt.grid(True)

# Gráfico 2: Sinal de Controle Real u(t)
plt.subplot(3, 1, 2)
plt.plot(t, u_bad, 'r--', label='Controle (Windup)')
plt.plot(t, u_good, 'b-', label='Controle (Back Calc)')
plt.hlines([u_limite], 0, T_sim, 'k', linestyles='dotted', label='Limite Saturação')
plt.ylabel('Sinal Aplicado u(t)')
plt.legend(loc='upper right')
plt.grid(True)

# Gráfico 3: Sinal Interno do Controlador v(t)
plt.subplot(3, 1, 3)
plt.plot(t, v_bad, 'r--', label='Sinal Calculado (Sem Correção)')
plt.plot(t, v_good, 'b-', label='Sinal Calculado (Corrigido)')
plt.hlines([u_limite], 0, T_sim, 'k', linestyles='dotted', label='Limite')
plt.ylabel('Sinal Interno v(t)')
plt.xlabel('Tempo (s)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

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