import numpy as np
import control as ct
import matplotlib.pyplot as plt
import re

t = np.linspace(0.0, 100.0, 1000)
# Questão 1:

# gráfico da equação 5:
tau1 = 3.0
tau2 = 1.0
k1 = 1
A = 2.0
y_5 = k1*A*(1 +(-tau1*np.exp(-t/tau1) - tau2*np.exp(-t/tau2))/(tau1 - tau2))

plt.plot(t, y_5, 'b', label='resposta para o sistema superamortecido')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

#gráfico da equação 6:
tau = 1.0
y_6 = k1*A*(1 - (1 - t/tau)*np.exp(-t/tau))
plt.plot(t, y_5, 'r', label='resposta para o sistema criticamente amortecido')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

#gráfico da equação 7 com o envoltório:

eps = 0.6
wd = 2.0
y_7 = k1*A*(1 - np.exp(-eps*t/tau)*(np.cos(wd*t)+(eps/(np.sqrt(1 - eps**2)) * np.sin(wd*t))))
env = k1*A*(1 - np.exp(-eps*t/tau))
plt.plot(t, y_7, 'g', label='resposta para o sistema subamortecido')
plt.plot(t, env, 'r', label='envoltório exponencial')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

#Questão 2:

g = ct.tf([10, 25], [1, 6, 25])
#usando a forma canônica, temos:
wn = 5.0
eps_g = 0.6
wd = wn*(np.sqrt(1 - eps_g))
tr = (np.pi - np.arccos(eps_g))/wd
tp = np.pi/wd
mp = 100.0*np.exp(-eps_g*np.pi/(np.sqrt(1 - eps_g**2)))

ts = 4/(eps_g*wn)

print(f'\n\nwn = {wn}\neps = {eps_g}\nwd = {wd}\ntr = {tr}\ntp = {tp}\nmp = {mp}\nts = {ts}\n\n')

# Questão 3:

dados = np.load('Y.npy')
dados2 = np.load('Y2.npy')
# print(f'\ndados 1: {dados}\n')
# print(f'\ndados 2: {dados2}\n')
t_3 = np.array([1.0 * i for i in range(len(dados))])
t_3_2 = np.array([1.0 * i for i in range(len(dados2))])

def rmse(y_r, y_m, n):
    return np.sqrt(1/(n) * np.sum((y_r - y_m) * (y_r - y_m)))

plt.plot(t_3, dados, 'g', label='dados das respostas de Y.npy')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t_3_2, dados2, 'g', label='dados das respostas de Y2.npy')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# Método Oldenbourg e Sartorius:

plt.plot(t_3, dados, 'g', label='dados das respostas de Y.npy')
plt.plot(t_3, 0.042611294*t_3 - 1.29979232, 'r', label='reta tangente ao ponto de inflexão')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# método de Sartorius
tc = 32.8
ta = 63.8
# como tc / ta < 0.735, o método de Sartorius não foi realizado

# método de Smith
a = 0.135
tb = 3.95
tf = ta - tc
th = 25.74
t1 = (tb + tf)*(1.0 - 200.0*(0.032 - a)*(1.0 - 0.086 + (0.0015/(0.032 - a))**(-1))**(-1.0))
t2 = tc - t1
sys_a1 = ct.tf([1], [t1, 1])
nun, den = ct.pade(th, 2)
atraso = ct.tf(nun, den)
sys_a = sys_a1 * atraso
k = np.mean(dados[210:320])
sys_k = ct.tf([k], [t2, 1])
sys_s = sys_a * sys_k
y_73 = k * 0.73
print(f'\nValor de 73% de y = {y_73}\n')
t, y_s = ct.step_response(sys_s, t_3)
plt.plot(t_3, dados, 'r', label='dados das respostas de Y.npy')
plt.plot(t_3, y_s, 'b', label='resposta do modelo pelo método de Smith')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()
atraso_m = atraso

#método de Sten:

ti = 64.2
th = ti + tc -ta - tb
nun, den = ct.pade(th, 2)
atraso = ct.tf(nun, den)
sys_a = sys_a1 * atraso
k = np.mean(dados[210:320])
sys_k = ct.tf([k], [t2, 1])
sys_st = sys_a * sys_k
sys_st = sys_a * sys_k

t, y_st = ct.step_response(sys_s, t_3)
plt.plot(t_3, dados, 'r', label='dados das respostas de Y.npy')
plt.plot(t_3, y_st, 'b', label='resposta do modelo pelo método de Sten')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# Método de Harriot:

t_73 = 87.595
y_026 = 0.26 * np.ones_like((t_3))
y_039 = 0.39 * np.ones_like((t_3))
t1_t2 = t_73/1.3
t_ht = 0.5 * t1_t2
print(f'intante t = {0.5 * t1_t2}')
y_t = 0.273
val_ht = 0.273/k
print(f'Validação de Harriot = {val_ht}')
# como o valor de y/KA não está na faixa desejada, esse método não é válido.

# Método de Meyer:

y_020 = 0.20 * k*  np.ones_like((t_3))
y_060 = 0.60 * k * np.ones_like((t_3))


plt.plot(t_3, dados, 'g', label='dados das respostas de Y.npy')
plt.plot(t_3, y_020, 'r', label='20% do valor máximo')
plt.plot(t_3, y_060, 'b', label='60% do valor máximo')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

t_60 = 45.71
t_20 = 14.60
tau_m = t_60/3.65
eps_m = 1.905
wn_m = 1.0/tau_m

sys_msa = ct.tf([k], [1/wn_m**2, 2*eps_m/wn_m, 1])      # para forçar que o ganho apareça sozinho no denominador, divide-se todo mundo por wn**2
sys_m = sys_msa * atraso_m      # considera-se o atraso de smith
y, y_m = ct.step_response(sys_m, t_3)

plt.plot(t_3, dados, 'r', label='dados das respostas de Y.npy')
plt.plot(t_3, y_m, 'b', label='resposta do modelo de Meyer')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

print(f'\nRMSE das respostas dos modelos desenvolvidos (dados 1):\n\n')
print(f'\nSmith : {rmse(dados, y_s, len(t_3))}\n')
print(f'\nSten : {rmse(dados, y_st, len(t_3))}\n')
print(f'\nMeyer : {rmse(dados, y_m, len(t_3))}\n')

# Questão 4 - validação dos dados:

ent = np.array([0.5 if i < 250 else 1.0 if i < 600 else 0.75 for i in range(len(dados2))])
t, y_s2 = ct.forced_response(sys_s, t_3_2, ent)
t, y_st2 = ct.forced_response(sys_st, t_3_2, ent)
t, y_m2 = ct.forced_response(sys_m, t_3_2, ent)

plt.plot(t_3_2, ent, 'b', label='entrada referente ao segundo conjunto de dados')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t_3_2, dados2, 'r', label='resposta real do sistema')
plt.plot(t_3_2, y_s2, 'b', label='resposta do modelo de Smith')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t_3_2, dados2, 'r', label='resposta real do sistema')
plt.plot(t_3_2, y_st2, 'b', label='resposta do modelo de Sten')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t_3_2, dados2, 'r', label='resposta real do sistema')
plt.plot(t_3_2, y_m2, 'b', label='resposta do modelo de Meyer')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

print(f'\nRMSE das respostas dos modelos desenvolvidos (dados 2):\n\n')
print(f'\nSmith : {rmse(dados2, y_s2, len(t_3_2))}\n')
print(f'\nSten : {rmse(dados2, y_st2, len(t_3_2))}\n')
print(f'\nMeyer : {rmse(dados2, y_m2, len(t_3_2))}\n')

# Questão 5 criação do modelo e validação deste:

with open("Dados_degrau_FanPlate.txt", "r") as f:
    data = f.read()

with open("Dados_degrau_FanPlate_2.txt", "r") as f2:
    data2 = f2.read()

pattern = r"Angle:\s*([-+]?\d*\.\d+|\d+),\s*Duty_cycle:\s*(\d+)%," \
          r"\s*Timer:\s*(\d+)ms"

pattern2 = r"Angle:\s*([-+]?\d*\.\d+|\d+),\s*Duty_cycle:\s*(\d+)%," \
          r"\s*Timer:\s*(\d+)ms"

matches = re.findall(pattern, data)
matches2 = re.findall(pattern2, data2)

angles = np.array([float(m[0]) for m in matches])
duty_cycles = np.array([int(m[1]) for m in matches])
timers = np.array([int(m[2]) for m in matches])

angles2 = np.array([float(m[0]) for m in matches2])
duty_cycles2 = np.array([int(m[1]) for m in matches2])
timers2 = np.array([int(m[2]) for m in matches2])

# como foi feito na prática anterior, a primeira parte da resposta foi descartada
t_ign = 15000
t_norm = timers[t_ign:] - t_ign
angles_segment = angles[t_ign:]

plt.plot(t_norm, angles_segment, 'b', label='dados da fanplate')
plt.xlabel('x')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# o modelo será baseado no ganho referente ao primeiro degrau, e no modelo de Meyer:

y0 = np.mean(angles_segment[:100])
u0 = np.mean(duty_cycles[13000:14000])

# valores após o regime permanente do degrau
y_ss = np.mean(angles_segment[1700:20000-t_ign]) 
u_ss = np.mean(duty_cycles[18000:25000]) 

# ganho incremental
k = (y_ss) / (u_ss)

y_020 = y0 + 0.20 * (y_ss - y0)*  np.ones_like((t_norm[37000:60000]))
y_060 = y0 + 0.60 * (y_ss - y0) * np.ones_like((t_norm[37000:60000]))

y_0 = np.mean(angles_segment[38000:43000])
y_ss = np.mean(angles_segment[19000:29000])

y_020 = y_0 + 0.20 * (y_ss - y_0)*  np.ones_like((t_norm[30000:]))
y_060 = y_0 + 0.60 * (y_ss - y_0) * np.ones_like((t_norm[30000:]))


plt.plot(t_norm[30000:], angles_segment[30000:], 'b', label='curva referente ao degrau 1')
plt.plot(t_norm[30000:], y_020, 'r', label='20% do valor máximo')
plt.plot(t_norm[30000:], y_060, 'g', label='60% do valor máximo')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

t_60_fp = 2830.0
t_20_fp = 1664.0

t_20_t_60 = 0.194774

t_60_tau = 3.35
eps_fp = 2.85
tau_fp = 844.776
wn_fp = 1/tau_fp

sys_fp = ct.tf([k], [1/wn_fp**2, 2*eps_fp/wn_fp, 1]) 

t, y_m_fp = ct.forced_response(sys_fp, timers, duty_cycles)
print(f'\nRMSE das respostas dos modelos desenvolvidos (dados da fan plate 1):\n\n')
print(f'\nMeyer : {rmse(angles, y_m_fp, len(timers))}\n')

t, y_m_fp2 = ct.forced_response(sys_fp, timers, duty_cycles2)
print(f'\nRMSE das respostas dos modelos desenvolvidos (dados da fan plate 2):\n\n')
print(f'\nMeyer : {rmse(angles, y_m_fp2, len(timers2))}\n')

plt.plot(timers, angles, 'r', label='dados 1 da fanplate')
plt.plot(timers, y_m_fp, 'b', label='dados do modelo')
plt.xlabel('x')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(timers, angles2, 'r', label='dados 2 da fanplate')
plt.plot(timers, y_m_fp2, 'b', label='dados do modelo')
plt.xlabel('x')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()





















