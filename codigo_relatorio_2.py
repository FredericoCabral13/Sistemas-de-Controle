import re
import numpy as np
import matplotlib.pyplot as plt
import control as ct

with open("Dados_degrau_FanPlate.txt"
, "r") as f:
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


# Método de Zigler Nichols:
t_ign = 15000
t_norm = timers[t_ign:] - t_ign
angle_op = angles[13000:15000].mean()
angles_segment = angles[t_ign:]
angle_norm = (angles_segment - angle_op) / np.max(np.abs(angles_segment - angle_op))

maxm = np.array([angle_norm[20000 - t_ign:25000 - t_ign].max()  for i in range(len(t_norm))])
minm = np.array([angle_norm[0] for i in range(len(t_norm))])

maxm2 = np.array([angle_norm[20000 - t_ign:30000 - t_ign].max() for i in range(len(t_norm))])
minm2 = np.array([angle_norm[35000 - t_ign:45000 - t_ign].min() for i in range(len(t_norm))])

maxm3 = np.array([angle_norm[35000 - t_ign:43000 - t_ign].max() for i in range(len(t_norm))])
minm3 = np.array([angle_norm[54000 - t_ign:58000 - t_ign].min() for i in range(len(t_norm))])

maxm4 = np.array([angle_norm[70000 - t_ign:75000 - t_ign].max() for i in range(len(t_norm))])
minm4 = np.array([angle_norm[54000 - t_ign:58000 - t_ign].min() for i in range(len(t_norm))])

def rmse(y_r, y_m):
    return np.sqrt(1/(len(timers) - t_ign) * np.sum((y_r - y_m) * (y_r - y_m)))

plt.plot(t_norm, angle_norm, 'b', label='normalização da resposta do sistema')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

plt.plot(t_norm[:20000-t_ign], angle_norm[:20000-t_ign], 'b', label='dados do degrau')
plt.plot(t_norm[:20000 - t_ign], 0.00117*t_norm[:20000 - t_ign] -0.64484, 'r')
plt.plot(t_norm[:20000-t_ign], maxm[:20000 - t_ign], 'g')
plt.plot(t_norm[:20000 - t_ign], minm[:20000 - t_ign], 'g')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

tau = 587.0
#k = (np.mean(angles[18000:25000]))/(np.mean(duty_cycles[18000:25000]))
# valores antes do degrau
y0 = np.mean(angles[13000:14000])
u0 = np.mean(duty_cycles[13000:14000])

# valores após o regime permanente do degrau
y_ss = np.mean(angles[18000:25000]) 
u_ss = np.mean(duty_cycles[18000:25000]) 

# ganho incremental
k = (y_ss) / (u_ss)
sys1 = ct.tf([k], [tau, 1])

plt.plot(t_norm[30000-t_ign:35000-t_ign], angle_norm[30000-t_ign:35000-t_ign], 'b', label='dados do degrau')
plt.plot(t_norm[30000-t_ign:35000-t_ign], maxm2[30000-t_ign:35000-t_ign], 'g')
plt.plot(t_norm[30000-t_ign:35000-t_ign], minm2[30000-t_ign:35000-t_ign], 'g')
plt.plot(t_norm[30000-t_ign:35000-t_ign], -0.000375*t_norm[30000-t_ign:35000-t_ign] + 6.7341, 'r')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

tau2 = 1228.0
#k2 = (np.mean(angles[35000:45000]))/(np.mean(duty_cycles[35000:45000]))
yss_2 = np.mean(angles[35000:43000])
uss_2 = np.mean(duty_cycles[35000:43000])
k2 = (yss_2) / (uss_2)
sys2 = ct.tf([k2], [tau2, 1])


plt.plot(t_norm[45000-t_ign:58000-t_ign], angle_norm[45000-t_ign:58000-t_ign], 'b', label='dados do degrau')
plt.plot(t_norm[45000-t_ign:58000-t_ign], maxm3[45000-t_ign:58000-t_ign], 'g')
plt.plot(t_norm[45000-t_ign:58000-t_ign], minm3[45000-t_ign:58000-t_ign], 'g')
plt.plot(t_norm[45000-t_ign:58000-t_ign], -0.00111*t_norm[45000-t_ign:58000-t_ign] + 33.913, 'r')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

tau3 = 1150.0
#k3 = np.mean(angles[55000:58000])/np.mean(duty_cycles[55000:58000])
yss_3 = np.mean(angles[55000:58000])
uss_3 = np.mean(duty_cycles[55000:58000])
k3 = (yss_3) / (uss_3)
sys3 = ct.tf([k3], [tau3, 1])


plt.plot(t_norm[60000-t_ign:75000-t_ign], angle_norm[60000-t_ign:75000-t_ign], 'b', label='dados do degrau')
plt.plot(t_norm[60000-t_ign:75000-t_ign], maxm4[60000-t_ign:75000-t_ign], 'g')
plt.plot(t_norm[60000-t_ign:75000-t_ign], minm4[60000-t_ign:75000-t_ign], 'g')
plt.plot(t_norm[60000-t_ign:75000-t_ign], 0.001786*t_norm[60000-t_ign:75000-t_ign] + -82.1881, 'r')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

tau4 = 980.0
#k4 = np.mean(angles[67000:75000])/ np.mean(duty_cycles[67000:75000])
yss_4 = np.mean(angles[70000:75000])
uss_4 = np.mean(duty_cycles[70000:75000])
k4 = (yss_4) / (uss_4)

sys4 = ct.tf([k4], [tau4, 1])


t, y1 = ct.forced_response(sys1, timers, duty_cycles2)
t, y2 = ct.forced_response(sys2, timers, duty_cycles2)
t, y3 = ct.forced_response(sys3, timers, duty_cycles2)
t, y4 = ct.forced_response(sys4, timers, duty_cycles2)



fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2 linhas x 2 colunas

# Primeiro degrau
axs[0, 0].plot(timers, y1, 'b', label='resposta do modelo ZN primeiro degrau')
axs[0, 0].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("y(t)")
axs[0, 0].legend()
axs[0, 0].grid()

# Segundo degrau
axs[0, 1].plot(timers, y2, 'b', label='resposta do modelo ZN segundo degrau')
axs[0, 1].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel("y(t)")
axs[0, 1].legend()
axs[0, 1].grid()

# Terceiro degrau
axs[1, 0].plot(timers, y3, 'b', label='resposta do modelo ZN terceiro degrau')
axs[1, 0].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("y(t)")
axs[1, 0].legend()
axs[1, 0].grid()

# Quarto degrau
axs[1, 1].plot(timers, y4, 'b', label='resposta do modelo ZN quarto degrau')
axs[1, 1].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[1, 1].set_xlabel("t")
axs[1, 1].set_ylabel("y(t)")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()  # ajusta espaçamento entre subplots
plt.show()

# método de Miller:
k_norm = np.mean(angle_norm[18000-t_ign:25000-t_ign])
k2_norm = np.mean(angle_norm[35000-t_ign:45000-t_ign])
k3_norm = np.mean(angle_norm[55000-t_ign:58000-t_ign])
k4_norm = np.mean(angle_norm[67000-t_ign:75000-t_ign])

array_k = np.array([k_norm for i in range(len(t_norm))])
array_k2 = np.array([k2_norm for i in range(len(t_norm))])
array_k3 = np.array([k3_norm for i in range(len(t_norm))])
array_k4 = np.array([k4_norm for i in range(len(t_norm))])

plt.plot(t_norm, angle_norm, 'b', label='resposta do sistema real')
plt.plot(t_norm, 0.63*array_k, 'r', label='63% do primeiro ganho')
plt.plot(t_norm, array_k2 + 0.37*(array_k - array_k2), 'g', label='63% do segundo ganho')
plt.plot(t_norm, array_k3 + 0.37*(array_k2 - array_k3), 'orange', label='63% do terceiro ganho')
plt.plot(t_norm, array_k3 + 0.63*(array_k4 - array_k3), 'purple', label='63% do quarto ganho')
plt.legend()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid()
plt.show()

plt.plot(t_norm[:20000-t_ign], angle_norm[:20000-t_ign], 'b', label='dados do degrau')
plt.plot(t_norm[:20000 - t_ign], 0.00111*t_norm[:20000 - t_ign] -0.59077, 'r')
plt.plot(t_norm[:20000-t_ign], maxm[:20000 - t_ign], 'g')
plt.plot(t_norm[:20000 - t_ign], minm[:20000 - t_ign], 'g')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

tau5 = 612.0
sys5 = ct.tf([k], [tau5, 1])

plt.plot(t_norm[25000-t_ign:35000-t_ign], angle_norm[25000-t_ign:35000-t_ign], 'b', label='dados do degrau')
plt.plot(t_norm[25000-t_ign:35000-t_ign], maxm2[25000-t_ign:35000-t_ign], 'g')
plt.plot(t_norm[25000-t_ign:35000-t_ign], minm2[25000-t_ign:35000-t_ign], 'g')
plt.plot(t_norm[25000-t_ign:35000-t_ign], -0.0000943*t_norm[25000-t_ign:35000-t_ign] + 2.01504, 'r')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

tau6 = 4829.0
sys6 = ct.tf([k2], [tau6, 1])

plt.plot(t_norm[40000-t_ign:58000-t_ign], angle_norm[40000-t_ign:58000-t_ign], 'b', label='dados do degrau')
plt.plot(t_norm[40000-t_ign:58000-t_ign], maxm3[40000-t_ign:58000-t_ign], 'g')
plt.plot(t_norm[40000-t_ign:58000-t_ign], minm3[40000-t_ign:58000-t_ign], 'g')
plt.plot(t_norm[40000-t_ign:58000-t_ign], -0.000261*t_norm[40000-t_ign:58000-t_ign] + 7.86284, 'r')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

tau7 = 4870.0
sys7 = ct.tf([k3], [tau7, 1])

plt.plot(t_norm[55000-t_ign:75000-t_ign], angle_norm[55000-t_ign:75000-t_ign], 'b', label='dados do degrau')
plt.plot(t_norm[55000-t_ign:75000-t_ign], maxm4[55000-t_ign:75000-t_ign], 'g')
plt.plot(t_norm[55000-t_ign:75000-t_ign], minm4[55000-t_ign:75000-t_ign], 'g')
plt.plot(t_norm[55000-t_ign:75000-t_ign], 0.00165*t_norm[55000-t_ign:75000-t_ign] + -75.90605, 'r')
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()

tau8 = 1041.0
sys8 = ct.tf([k4], [tau8, 1])

t, y5 = ct.forced_response(sys5, timers, duty_cycles2)
t, y6 = ct.forced_response(sys6, timers, duty_cycles2)
t, y7 = ct.forced_response(sys7, timers, duty_cycles2)
t, y8 = ct.forced_response(sys8, timers, duty_cycles2)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2 linhas x 2 colunas

# Primeiro degrau
axs[0, 0].plot(timers, y5, 'b', label='resposta do modelo de Miller primeiro degrau')
axs[0, 0].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("y(t)")
axs[0, 0].legend()
axs[0, 0].grid()

# Segundo degrau
axs[0, 1].plot(timers, y6, 'b', label='resposta do modelo de Miller segundo degrau')
axs[0, 1].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel("y(t)")
axs[0, 1].legend()
axs[0, 1].grid()

# Terceiro degrau
axs[1, 0].plot(timers, y7, 'b', label='resposta do modelo de Miller terceiro degrau')
axs[1, 0].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("y(t)")
axs[1, 0].legend()
axs[1, 0].grid()

# Quarto degrau
axs[1, 1].plot(timers, y8, 'b', label='resposta do modelo de Miller quarto degrau')
axs[1, 1].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[1, 1].set_xlabel("t")
axs[1, 1].set_ylabel("y(t)")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()  # ajusta o espaçamento entre os subplots
plt.show()

# Método de Smith:

dy = np.mean(angles[18000:25000])
p_1 = np.array([angles[t_ign] + 0.284 * (dy - angles[t_ign]) for i in range(len(t_norm))])
p_2 = np.array([angles[t_ign] + 0.632 * (dy - angles[t_ign]) for i in range(len(t_norm))])

dy2 = np.mean(angles[35000:45000])
p_12 = np.array([dy2 + 0.456 * (dy - dy2) for i in range(len(t_norm))])
p_22 = np.array([dy2 + 0.789 * (dy - dy2) for i in range(len(t_norm))])

dy3 = np.mean(angles[55000:58000])
p_13 = np.array([dy3 + 0.456 * (dy2 - dy3) for i in range(len(t_norm))])
p_23 = np.array([dy3 + 0.789 * (dy2 - dy3) for i in range(len(t_norm))])

dy4 = np.mean(angles[67000:75000])
p_14 = np.array([dy3 + 0.456 * (dy4 - dy3) for i in range(len(t_norm))])
p_24 = np.array([dy3 + 0.789 * (dy4 - dy3) for i in range(len(t_norm))])

plt.plot(t_norm, angles[t_ign:], 'b', label='dados do sistema')
plt.plot(t_norm, p_1, 'g')
plt.plot(t_norm, p_2, 'g')
plt.plot(t_norm, p_12, 'r')
plt.plot(t_norm, p_22, 'r')
plt.plot(t_norm, p_13, 'purple')
plt.plot(t_norm, p_23, 'purple')
plt.plot(t_norm, p_14, 'black')
plt.plot(t_norm, p_24, 'black')
plt.legend()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid()
plt.show()

tau9 = 238.5
sys9 = ct.tf([k], [tau9, 1])
tau10 = 1152.0
sys10 = ct.tf([k2], [tau10, 1])
tau11 = 1950.0
sys11 = ct.tf([k3], [tau11, 1])
tau12 = 624.0
sys12 = ct.tf([k4], [tau12, 1])

t, y9 = ct.forced_response(sys9, timers, duty_cycles2)
t, y10 = ct.forced_response(sys10, timers, duty_cycles2)
t, y11 = ct.forced_response(sys11, timers, duty_cycles2)
t, y12 = ct.forced_response(sys12, timers, duty_cycles2)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2 linhas x 2 colunas

# Primeiro degrau
axs[0, 0].plot(timers, y9, 'b', label='resposta do modelo de Smith primeiro degrau')
axs[0, 0].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("y(t)")
axs[0, 0].legend()
axs[0, 0].grid()

# Segundo degrau
axs[0, 1].plot(timers, y10, 'b', label='resposta do modelo de Smith segundo degrau')
axs[0, 1].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel("y(t)")
axs[0, 1].legend()
axs[0, 1].grid()

# Terceiro degrau
axs[1, 0].plot(timers, y11, 'b', label='resposta do modelo de Smith terceiro degrau')
axs[1, 0].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("y(t)")
axs[1, 0].legend()
axs[1, 0].grid()

# Quarto degrau
axs[1, 1].plot(timers, y12, 'b', label='resposta do modelo de Smith quarto degrau')
axs[1, 1].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[1, 1].set_xlabel("t")
axs[1, 1].set_ylabel("y(t)")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()  # ajusta o espaçamento entre os subplots
plt.show()

# Método de Krishnaswamy:

tau13 = 107.325
sys13 = ct.tf([k], [tau13, 1])
tau14 = 518.4
sys14 = ct.tf([k2], [tau14, 1])
tau15 = 877.5
sys15 = ct.tf([k3], [tau15, 1])
tau16 = 280.8
sys16 = ct.tf([k4], [tau16, 1])

t, y13 = ct.forced_response(sys13, timers, duty_cycles2)
t, y14 = ct.forced_response(sys14, timers, duty_cycles2)
t, y15 = ct.forced_response(sys15, timers, duty_cycles2)
t, y16 = ct.forced_response(sys16, timers, duty_cycles2)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2 linhas x 2 colunas

# Primeiro degrau
axs[0, 0].plot(timers, y13, 'b', label='resposta do modelo de Sundaresan e Krishnaswamy primeiro degrau')
axs[0, 0].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("y(t)")
axs[0, 0].legend()
axs[0, 0].grid()

# Segundo degrau
axs[0, 1].plot(timers, y14, 'b', label='resposta do modelo de Sundaresan e Krishnaswamy segundo degrau')
axs[0, 1].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel("y(t)")
axs[0, 1].legend()
axs[0, 1].grid()

# Terceiro degrau
axs[1, 0].plot(timers, y15, 'b', label='resposta do modelo de Sundaresan e Krishnaswamy terceiro degrau')
axs[1, 0].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("y(t)")
axs[1, 0].legend()
axs[1, 0].grid()

# Quarto degrau
axs[1, 1].plot(timers, y16, 'b', label='resposta do modelo de Sundaresan e Krishnaswamy quarto degrau')
axs[1, 1].plot(timers, angles2, 'r', label='resposta do sistema real')
axs[1, 1].set_xlabel("t")
axs[1, 1].set_ylabel("y(t)")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()  # ajusta o espaçamento entre subplots
plt.show()

resp = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16]

for i in range(16):
    if(i < 4):
        print(f'\nRMSE do modelo de Zigler Nichols (degrau {i + 1}) : {rmse(angles2, resp[i])}\n')
    elif(i < 8):
        print(f'\nRMSE do modelo de Miller (degrau {i + 1 - 4}) : {rmse(angles2, resp[i])}\n')
    elif(i < 12):
        print(f'\nRMSE do modelo de Smith (degrau {i + 1 - 8}) : {rmse(angles2, resp[i])}\n')
    elif(i < 16):
        print(f'\nRMSE do modelo de Sundaresan e Krishnaswamy (degrau {i + 1 - 12}) : {rmse(angles2, resp[i])}\n')





















