try:
    from sippy_unipi import system_identification
except ImportError:
    import os
    import sys
    sys.path.append(os.pardir)
    from sippy_unipi import system_identification

import control.matlab as cnt
import matplotlib.pyplot as plt
import numpy as np
from tf2ss import lsim
from sippy_unipi import functionset as fset
import control as ct

var_list = [0.001]  # Variância do ruído. Medida do quanto os valores de ruído se desviam da média (0 para o ruído branco)
ts = 1.0            # Tempo de amostragem          

G1 = cnt.tf([1], [1, 4, 6, 4, 1])  # Função de transferência do sistema que irá gerar a saída segundo a entrada conhecida. Nesse tipo de identificação, a função de transferência do sistema não é conhecida
H = cnt.tf([1, 0, 0], [1, 1, 1])  # Função de transferência do filtro que passará o ruido branco para ruìdo colorido (possui preponderância em determinadas frequências)
g21 = cnt.tf([1], [1, 3])
g22 = cnt.tf([1, 5], [1, 1, 1])
G2 = g21 * g22

na = 1      # Ordem do polinômio do denominador
nb = 1      # Ordem do polinômio do numerador
th = 0      # Valores relacionados aos atrasos

na2 = 2
nb2 = 2
th2 = 0

print(f'\nFunção de transferência do sistema:\n{G1}\n')

tfin = 400                  # Último valor de tempo
npts = int(tfin / ts) + 1   # Número de intervalos de tempo
Time = np.linspace(0, tfin, npts)   # Array de tempo considerando o passso o número de pontas e do tempo final

Usim, _, _ = fset.GBN_seq(npts, 0.1, Range=[0, 1.0])    # Geração de uma entrada para a identificação do sistema, para que o sistema responda segundo essa entrada, e seja possível fazer a identificação
Usim = Usim.flatten()           # Garantindo que Usim é um array unidimensional

err_inputH = np.random.normal(0, np.sqrt(var_list[0]), npts)     # Geração de um ruído branco a partir de uma distribuição normal aleatória   
err_outputH, Time, _ = lsim(H, err_inputH, Time)                # Filtragem do ruido branco, gerando ruido colorido

Time, Yout = ct.forced_response(G1, Time, Usim)                 # Resposta do sistema a partir da entrada gerada
Ytot = Yout + err_outputH  # saída total (entrada gerada + ruído)

Id_ARX = system_identification(                 # Função que retorna um objeto que contém os dados de identificação do sistema, como a saída do modelo tendo em vista os parÂmetros passados
    Ytot, Usim, "ARX", ARX_orders=[na, nb, th]  # Essa função recebe como parâmetros a entrada gerada e a saída real do sistema, as ordens do numerador e do denominador do modelo buscado
)                                               # Essa função a partir dos dados de entrada e de saída e das características do modelo a ser identificado (ordens e atraso), faz com que o vetor de coeficientes theta seja descoberto e, realiza a otimização desse vetor de forma que a soma dos resíduos seja minimizada, tendo em vista que que a filtragem do ruido depende de parÂmtros do modelo (poninômio A)

Yout_ARX = Id_ARX.Yid                       # Saída do modelo identificado
Yout_ARX = Yout_ARX.flatten()               # Garantindo que Yout_ARX é um array unidimensional

Time2, Yout2 = ct.forced_response(G2, Time, Usim)                 # Resposta do sistema a partir da entrada gerada
Ytot2 = Yout2 + err_outputH  # saída total (entrada gerada + ruído)

Id_ARX2 = system_identification(                 # Função que retorna um objeto que contém os dados de identificação do sistema, como a saída do modelo tendo em vista os parÂmetros passados
    Ytot2, Usim, "ARX", ARX_orders=[na2, nb2, th2]  # Essa função recebe como parâmetros a entrada gerada e a saída real do sistema, as ordens do numerador e do denominador do modelo buscado
) 

Yout_ARX2 = Id_ARX2.Yid                       # Saída do modelo identificado
Yout_ARX2 = Yout_ARX2.flatten()               # Garantindo que Yout_ARX é um array unidimensional

B = Id_ARX.NUMERATOR            # Array de coeficientes do numerador para o modelo identificado    
A = Id_ARX.DENOMINATOR          # Array de coeficientes do denominador para o modelo identificado      
G_i = ct.tf(B, A, ts)           # Função de transferência do modelo. Ela ligeiramente para cada execução do programa, mas continua gerando um resultado muito bom de acordo com a ordem dos modelos

print(f'\nFunção de transferência do modelo identificado:\n {G_i}')

B2 = Id_ARX2.NUMERATOR
A2 = Id_ARX2.DENOMINATOR
G_i2 = ct.tf(B2, A2, ts)

print(f'\nFunção de transferência do modelo identificado:\n {G_i2}')

def rmse(y_r, y_m, n):          # Função que determina o RMSE, índice de validação do modelo obtido. Essa função consiste na obtenção do módulo da diferença entre as integrais (somatórias) das resposta real e do modelo
    return np.sqrt(1/(n) * np.sum((y_r - y_m) * (y_r - y_m)))

print(f'\n\nValor do RMSE considerando o modelo buscado para G1 : {rmse(Ytot, Yout_ARX, len(Time))}\n')
print(f'\nValor do RMSE considerando o modelo buscado para G2 : {rmse(Ytot2, Yout_ARX2, len(Time))}\n')

# plotagem dos gráficos:

plt.plot(Time, Usim, label='entrada')
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.grid()
plt.show()

plt.plot(Time, Ytot , label='saida real')
plt.plot(Time, Yout_ARX ,'r', label='saida do modelo')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

plt.plot(Time, Ytot2 , label='saida real G2')
plt.plot(Time, Yout_ARX2 ,'r', label='saida do modelo G2')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

plt.plot(Time, err_inputH, label='ruido')
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.grid()
plt.show()






