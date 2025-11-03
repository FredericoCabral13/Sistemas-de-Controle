import numpy as np
import control as ct
import matplotlib.pyplot as plt
import re

t = np.linspace(0.0, 100.0, 1000)

dados = np.load('Y.npy')
dados2 = np.load('Y2.npy')
print(f'\ndados 1: {dados}\n')
print(f'\ndados 2: {dados2}\n')
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


# método de Smith

tc = 32.8
ta = 63.8
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

# Método de Meyer:

y_020 = 0.20 * k*  np.ones_like((t_3))
y_060 = 0.60 * k * np.ones_like((t_3))

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


print(f'Função de transferência do modelo pelo método de Meyer : \n{sys_m }\n')    # Meyer
print(f'Função de transferência do modelo pelo método de Sten : \n{sys_st }\n')   # Sten
print(f'Função de transferência do modelo pelo método de Smith : \n{sys_s }\n')    # Smith 


#testando as respostas a partir da aplicação de um ganho proporcional kp:

kp = ct.tf([1.161], [1.0])
G_cl = ct.feedback((sys_m * kp), 1)
t_3, y_Gcl = ct.step_response(G_cl, t_3)
kcr1 = 1.161
pcr1 = 95.8
kp1 = 0.6 * kcr1
ti1 = 0.5 * pcr1
td1 = 0.125*pcr1
PID1 = ct.tf([kp1*td1, kp1, kp1/ti1], [1.0, 0.0])
G_cl_pid = ct.feedback((sys_m * PID1), 1)
t_3, y_Gcl_pid = ct.step_response(G_cl_pid, t_3)

kp2 = ct.tf([0.723], [1.0])
G_cl2 = ct.feedback((sys_st * kp2), 1)
t_3, y_Gcl2 = ct.step_response(G_cl2, t_3)
kcr2 = 0.720
pcr2 = 106.5
kp2 = 0.6 * kcr2
ti2 = 0.5 * pcr2
td2 = 0.125*pcr2
PID2 = ct.tf([kp2*td2, kp2, kp2/ti2], [1.0, 0.0])
G_cl_pid2 = ct.feedback((sys_m * PID2), 1)
t_3, y_Gcl_pid2 = ct.step_response(G_cl_pid2, t_3)

kp3 = ct.tf([0.779], [1.0])
G_cl3 = ct.feedback((sys_s * kp3), 1)
t_3, y_Gcl3 = ct.step_response(G_cl3, t_3)
kcr3 = 0.770
pcr3 = 97.5
kp3 = 0.6 * kcr3
ti3 = 0.5 * pcr3
td3 = 0.125 * pcr3
PID3 = ct.tf([kp3*td3, kp3, kp3/ti3], [1.0, 0.0])
G_cl_pid3 = ct.feedback((sys_m * PID3), 1)
t_3, y_Gcl_pid3 = ct.step_response(G_cl_pid3, t_3)


# Respostas para a malha fechada usando os PIDs projetados:

plt.plot(t_3, y_Gcl, 'b', label='resposta do modelo de Meyer')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t_3, y_Gcl2, 'b', label='resposta do modelo de Sten')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t_3, y_Gcl3, 'b', label='resposta do modelo de Smith')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# Respostas para a malha fechada usando os PIDs projetados:

plt.plot(t_3, y_Gcl_pid, 'b', label='resposta do modelo de Meyer (PID)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t_3, y_Gcl_pid2, 'b', label='resposta do modelo de Sten (PID)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

plt.plot(t_3, y_Gcl_pid3, 'b', label='resposta do modelo de Smith (PID)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

t = np.linspace(0.0, 30.0, 2000)      # tempo de simulação
ref = np.ones_like(t)                 # referência degrau unitário

kcr4 = 39.8
planta = ct.tf([kcr4*2.709, kcr4*0.6316 , kcr4*0.04907], [206.2, 80.86, 12.38, 0.8272 , 0.01811], name="planta", inputs="u", outputs="y")
sensor = ct.tf([1], [1], name="sensor", inputs="u", outputs="y")
sum_err = ct.summing_junction(inputs=["r", "-ysensor"], output="e", name="sum_err")

# === Controlador On-Off (não linear) ===
def out_fcn(t, x, u, params=None):
    # u[0] = erro
    if u[0] > 0:
        return [1.5]
    elif u[0] < 0:
        return [-1.5]
    else:
        return [0.0]

def updfcn(t, x, u, params=None):
    return []  # sem estados internos

on_off = ct.NonlinearIOSystem(
    updfcn=updfcn,
    outfcn=out_fcn,
    inputs=1,
    outputs=1,
    states=0,
    name="on_off"
)

# === Interconexão dos blocos ===
malha_f = ct.interconnect(
    syslist=[planta, sensor, sum_err, on_off],
    connections=[
        ['on_off.u', 'sum_err.e'],
        ['planta.u', 'on_off.y'],
        ['sensor.u', 'planta.y'],
        ['sum_err.ysensor', 'sensor.y']
    ],
    inplist=["r"],
    outlist=["planta.y", "sum_err.e", "on_off.y"]
)

# === Simulação com input_output_response (não linear) ===
t_out, y_out = ct.input_output_response(malha_f, T=t, U=ref)

# Separar as saídas
y = y_out[0, :]   # saída da planta
e = y_out[1, :]   # erro
u = y_out[2, :]   # sinal de controle (saída do relé)

# === Plotagem ===
plt.figure(figsize=(10,6))
plt.plot(t, ref, 'k--', label="Referência (r)")
plt.plot(t, y, 'r', label="Saída (y)")
plt.step(t, u, 'b', label="Controle (u)", where="post")
plt.plot(t, e, 'g', alpha=0.7, label="Erro (e)")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title("Controle On-Off em Malha Fechada")
plt.legend()
plt.grid(True)
plt.show()

# (opcional) Imprimir primeiros valores
for i in range(0, len(t), 200):
    print(f"t={t[i]:.2f}s | r={ref[i]:.2f} | y={y[i]:.3f} | e={e[i]:.3f} | u={u[i]:.3f}")


#Controlador de 1a ordem

# Parâmetros da planta
K = 2.7
L = 32.7
T = 48.6

# Função de transferência sem o atraso
G = ct.tf([K], [T, 1])

num_pade, den_pade = ct.pade(L, 1)
Pade = ct.tf(num_pade, den_pade)
G_pade = G * Pade

#Controlador 1: PID (Ziegler-Nichols)
Kp = T / (K * L)
Ti = 2 * L
Td = 0.5 * L

#filtro derivativo
s = ct.tf('s')
N_filtro = 10
Tf = Td / N_filtro

C_p = Kp
C_i = Kp / (Ti * s)
C_d = (Kp * Td * s) / (Tf * s + 1)
C = C_p + C_i + C_d

# Malha aberta e fechada (Controlador 1)
L_open = C * G_pade
T_closed = ct.feedback(L_open, 1)


#Controlador 2 (Minimização do ITAE)
Kp2 = (0.965 / K) * (L / T)**(-0.850)
Ti2 = (T / 0.796) * (T / L)**(-0.147)
Td2 = (0.308 * T) * (L / T)**(0.929)

# Aplicando o mesmo filtro derivativo
Tf2 = Td2 / N_filtro

# Construindo o PID 2 (paralelo, filtrado)
C2_p = Kp2
C2_i = Kp2 / (Ti2 * s)
C2_d = (Kp2 * Td2 * s) / (Tf2 * s + 1)
C2 = C2_p + C2_i + C2_d

# Malha aberta e fechada (Controlador 2)
L_open2 = C2 * G_pade
T_closed2 = ct.feedback(L_open2, 1)


t_sim = np.arange(len(dados))

# Controlador 1
t_out, y_unitario = ct.step_response(T_closed, T=t_sim)
# Controlador 2
t_out2, y_unitario2 = ct.step_response(T_closed2, T=t_sim)


plt.figure(figsize=(10, 6))
plt.plot(t_sim, dados, 'bo', alpha=0.5, label=f'Dados Y.npy (Experimental, Ganho={K})', markersize=4)

#Correção do offset
y_simulado_corrigido = y_unitario * K
y_simulado_corrigido2 = y_unitario2 * K

#RMSE
erro_1 = y_simulado_corrigido - dados
mse_1 = np.mean(erro_1**2)
rmse_1 = np.sqrt(mse_1)

erro_2 = y_simulado_corrigido2 - dados
mse_2 = np.mean(erro_2**2)
rmse_2 = np.sqrt(mse_2)



label_c1 = f'PID - Ziegler-Nichols (RMSE = {rmse_1:.4f})'
plt.plot(t_out, y_simulado_corrigido, 'r-', linewidth=2, label=label_c1)

label_c2 = f'PID - Minimização do ITAE (RMSE = {rmse_2:.4f})'
plt.plot(t_out2, y_simulado_corrigido2, 'g-', linewidth=2, label=label_c2)


plt.xlabel('Tempo (s)')
plt.ylabel('y(t)')
plt.title('Comparação: Dados Experimentais vs. Simulação PID')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


print("\n--- Controlador 1: PID (Ziegler-Nichols) ---")
print(f"Kp = {Kp:.5f}")
print(f"Ti = {Ti:.2f} s")
print(f"Td = {Td:.2f} s")
print(f"Tf (Filtro) = {Tf:.3f} s")
print(f"RMSE vs Dados = {rmse_1:.5f}")
print("\nFunção de Transferência do Controlador C(s):")
print(C)
print("\nFunção de Transferência de Malha Fechada T(s):")
print(T_closed)

print("\n\n--- Controlador 2: Minimização do ITAE ---")
print(f"Kp = {Kp2:.5f}")
print(f"Ti = {Ti2:.2f} s")
print(f"Td = {Td2:.2f} s")
print(f"Tf (Filtro) = {Tf2:.3f} s")
print(f"RMSE vs Dados = {rmse_2:.5f}")
print("\nFunção de Transferência do Controlador C2(s):")
print(C2)
print("\nFunção de Transferência de Malha Fechada T2(s):")
print(T_closed2)


print("\n\n--- Validação RMSE (Comparação com Dados Experimentais) ---")
if rmse_1 < rmse_2:
    print(f"Conclusão: O Controlador 1 (Ziegler-Nichols) tem a saída mais próxima dos dados experimentais (RMSE: {rmse_1:.5f} < {rmse_2:.5f})")
elif rmse_2 < rmse_1:
    print(f"Conclusão: O Controlador 2 (Minimização do ITAE) tem a saída mais próxima dos dados experimentais (RMSE: {rmse_2:.5f} < {rmse_1:.5f})")
else:
    print(f"\nConclusão: Ambos os controladores têm um RMSE idêntico.")































