import numpy as np
import control as ct
import matplotlib.pyplot as plt

g_1 = ct.tf([1], [1, 0])
g_2 = ct.tf([1], [1, 1])
g_3 = ct.tf([1], [1,  5])

G = g_1*g_2*g_3

# Topologia em paralelo:
Kp = 18.0
kp = ct.tf([Kp], [1])
ti = 1.405
td = 0.35125

c_p = ct.tf([1], [1])
c_i = ct.tf([1], [ti, 0])
c_d = ct.tf([td, 0], [1])

C = kp*(c_p + c_i + c_d)
H = ct.tf([1], [1])

print(f'\n Função de transferência da planta G(s):\n {G}')
print(f'\n Função de transferência do controldor C(s):\n {C}')

G_cl = ct.feedback((G*C), H)

t = np.linspace(0.0, 100.0, 1000)

t, y = ct.step_response(G_cl, t)

# Topologia em série:

ti_s = ti/2 * (1 + np.sqrt(1 - 4*td/ti))
td_s = ti/2 * (1 - np.sqrt(1 - 4*td/ti))
Kp_s = Kp/(1 + td_s/ti_s)

kp_s = ct.tf([Kp_s], [1])
termo_d = ct.tf([td_s, 1], [1])
c_p_s = ct.tf([1], [1])
c_i_s = ct.tf([1], [ti_s, 0])

C2 = termo_d*kp_s*(c_p_s + c_i_s)

G_s_cl = ct.feedback((G*C2), H)

t, y_s = ct.step_response(G_s_cl, t)

plt.plot(t, y_s, 'r', label='topologia em série')
plt.plot(t, y, 'b--', label='topologia em paralelo')
# plt.title('gráfico da resposta do sistema em malha fechada para o PID em série')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()
