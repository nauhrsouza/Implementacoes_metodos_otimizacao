#bibliotecas

import sympy as sp
from math import*
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
eps=1E-6

"""#Gradiente projetado função Bazaraa

##Função exemplo

def funcao(x):
  return x[0]**2 + x[1]**2 - 2*x[0] - 4*x[1]

def gradiente(x):
  return np.array([2*x[0]-2,
          2*x[1]-4])

def restricao(x):
  return np.array([
    x[0]+x[1],
    x[0],
    x[1]
  ])

def matriz_restricao():
    return [[1,1],
            [-1,0],
            [0,-1]]

def vet_b():
  return [4,0,0]

def g1(x):
  return (4-x)
"""

def funcao(x):
  return x[0]**2 + x[1]**2 +x[2]**2 +x[3]**2-2*x[0]-3*x[3]


def gradiente(x):
  return [2*x[0]-2,2*x[1],2*x[2],2*x[3]-3]


def restricao(x):
  return np.array([
      2*x[0]+x[1]+x[2]+4*x[3],
      x[0]+x[1]+2*x[2]+x[3],
      -x[0],-x[1],-x[2],-x[3]],dtype=float)

def matriz_restricao():
    return np.array(
          [[2,1,1,4],
            [1,1,2,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1]],float)

def vet_b():
  return  np.array([7,6,0,0,0,0],float)

def g1(x):
  return  (7-2*x[0]-x[1]-x[2])/4

def g2(x):
  return 6-x[0]-x[1]-2*x[2]

"""def funcao(x):
  return 2*x[0]**2 + 2*x[1]**2 -2*x[0]*x[1]-4*x[0]-6*x[1]


def gradiente(x):
  return [4*x[0]-2*x[1]-4,
          4*x[1]-2*x[0]-6]


def restricao(x):
  return np.array([
      x[0] + x[1],
      x[0] + 5 * x[1],
      -x[0],
      -x[1]],dtype=float)

def matriz_restricao():
    return np.array([[1,1],
            [1,5],
            [-1,0],
            [0,-1]],float)

def vet_b():
  return  np.array([2,5,0,0],float)

def g1(x):
  return 2-x

def g2(x):
  return (5-x)/5

##Funções usadas no método
"""

def inv(M):
  if np.ndim(M)==0:
    return M
  else:
    return np.linalg.inv(M)

def restricoes_ativas(x):
  ativas=[]
  for i in range(len(restricao(x))):
    if np.linalg.norm(restricao(x)[i] - vet_b()[i])<=eps:
      ativas.append(matriz_restricao()[i])
  return ativas

def vet_ativos(x):
  vet_ativas=[]
  for i in range(len(restricao(x))):
    if np.linalg.norm(restricao(x)[i]- vet_b()[i])<=eps:
      vet_ativas.append(vet_b()[i])
  return vet_ativas

def restricoes_inativas(x):
  inativas=[]
  for i in range(len(restricao(x))):
    if np.linalg.norm(restricao(x)[i]- vet_b()[i])>eps:
      inativas.append(matriz_restricao()[i])
  return inativas

def vet_inativos(x):
  vet_inativas=[]
  for i in range(len(restricao(x))):
    if np.linalg.norm(restricao(x)[i]- vet_b()[i])>eps:
      vet_inativas.append(vet_b()[i])
  return vet_inativas

def projecao(mat):
    return np.round(np.eye(len(mat[0])) - np.dot(np.array(mat).T,np.dot(inv(np.dot(mat,np.array(mat).T)),mat)),6)

def direcao(x,mat):
    if restricoes_ativas(x)!=[]:
      return np.round(-np.dot(projecao(mat),gradiente(x)),6)
    else:
      if np.all(gradiente(x)==0):
        print("Não há restrições ativas")
      else:
        return np.dot(-1,gradiente(x))


def lambda_k(x,mat):
    l=np.array([],dtype=float)
    b=vet_inativos(x)-np.dot(restricoes_inativas(x),x)
    print(b)
    d=np.dot(restricoes_inativas(x),direcao(x,mat))
    print(d)
    for i in range(len(b)):
      if d[i]>0:
        l = np.append(b[i]/d[i],l)
      else:
        np.append(0,l)

    return np.min(l)

def multiplicador_lagrange(mat,grad):
    mat = np.array(mat)
    return np.round(-np.dot(np.dot(inv(np.dot(mat,mat.T)),mat),grad),6)

def deleta_linha(mat,grad):
    mat = np.delete(mat,multiplicador_lagrange(mat,grad).argmin(),0)
    return mat

"""##Buscas

###Busca da Seção áurea
"""

eps = 1E-5
bMax = 10**8
theta1 = (3-sqrt(5))/2
theta2 = 1-theta1
def phi_t(t,x,d):
    t = x + np.dot(t,d)
    phi =  funcao(t)
    return phi
def aurea(x,d,rho):
    
    a = 0
    s = rho
    b = 2*rho
    phib = phi_t(b, x,d)
    phis = phi_t(s, x,d)
    if phib < phis and 2*b < bMax:
        a = s
        s = b
        b = 2*b
        phis = phib
        phib = phi_t(b, x,d)
   
    u = a + theta1 * (b - a)
    v = a + theta2 * (b - a)
    phiu = phi_t(u, x,d)
    phiv = phi_t(v, x,d)
    if (b-a) > eps:
        if phiu < phiv:
            b = v
            v = u
            u = a + theta1 * (b - a)
            phiv = phiu
            phiu = phi_t(u, x,d)
        else:
            a = u
            u = v
            v = a + theta2 * (b - a)
            phiu = phiv
            phiv = phi_t(v, x,d)
    t_k = (u+v)/2
    return  t_k

"""###Busca de Armijo"""

def armijo(x,d,l=float, gama=0.5,  eta= 0.1):
  l_min=eps
  gd = np.dot(gradiente(x), d)
  if gd > 0:
    return 0
  f = funcao(x)
  while l > l_min:
    f_t =funcao(x + np.dot(l,d))
    if f_t <= f+ eta*l*gd:
        return l
    l *= gama

  return l

"""##Método"""

def metodo(x,m,recursion_depth=0):  
  max_recursion_depth = 1000  
  #if restricoes_ativas(x)!=[]:
  grad_k.append(np.round(gradiente(x),5))
  d_k.append(direcao(x, m))
  a_1.append(m)
  proj.append(projecao(m))
  kkt.append(multiplicador_lagrange(m, gradiente(x)))

  if np.linalg.norm(direcao(x, m)) > eps:

    projecao_direcao = np.dot(projecao(m), direcao(x, m))

    l_max = lambda_k(x, m)
    print("max",l_max)
    l_k = armijo(x, direcao(x, m), l_max)
    print("lk",l_k)
    l.append(l_k)

    x = x + np.dot(l_k, direcao(x, m))
    p_k.append(x)
  print("d",direcao(x,m))
  f_x.append(funcao(x))
  if np.min(multiplicador_lagrange(m, gradiente(x))) >= eps:
    return x
  else:
    m = deleta_linha(restricoes_ativas(x), gradiente(x))
    if len(m) == 0:
      
      if all(restricao(x) <= 0) and recursion_depth < max_recursion_depth:  
        return metodo(x, restricoes_ativas(x), recursion_depth + 1)
      else:
        print("Maximum recursion depth reached or infeasible point. Returning current solution.")
        return x
    else:
      if recursion_depth < max_recursion_depth:
        return metodo(x, m, recursion_depth + 1)
      else:
        print("Maximum recursion depth reached. Returning current solution.")
        return x

"""#Executável"""

#x=[0,0]
x=[2,2,1,0]
x_k=np.array(x,float)
p_k=[]
d_k=[]
grad_k=[]
f_x=[]
a_1=[]
proj=[]
kkt=[]
l=[]
p_k.append(x_k)
f_x.append(funcao(x_k))
l.append(0)
metodo(x_k,restricoes_ativas(x_k))

"""##df exemplo 2"""

df1=pd.DataFrame(data = {'x1':np.array(p_k)[:,0], 'x2':np.array(p_k)[:,1],'x3':np.array(p_k)[:,2], 'x4':np.array(p_k)[:,3],'lambda':l})
df1

df1.to_latex(index=False)

df2=pd.DataFrame(data = {'Restrições ativas':a_1,'Projeção':proj}, index = pd.Index([i for i in range(len(proj))]))
df2

df2.to_latex(index=False)

df3=pd.DataFrame(data = {'Restrições ativas':a_1,'Projeção':proj ,'Gradiente':grad_k,'Direção':d_k,'Multiplicadores':kkt})
df3

df3.to_latex(index=False)

df4=pd.DataFrame(data = {'f(x1,x2,x3,x4)':f_x})
df4

df4.to_latex(index=False)

"""##dfs default"""

df1=pd.DataFrame(data = {'x1':np.array(p_k)[:,0], 'x2':np.array(p_k)[:,1],'f(x1,x2)':f_x, 'Gradiente':grad_k,'Direção':d_k,'Multiplicadores':kkt,'lambda':l}, index = pd.Index([i for i in range(len(p_k))]))
df1

df2=pd.DataFrame(data = {'Restrições ativas':a_1,'Projeção':proj}, index = pd.Index([i for i in range(len(proj))]))
df2

"""#Gerando graficos"""

px=[]
py=[]
for i in p_k:
  px.append(i[0])
  py.append(i[1])

dx=[]
dy=[]
for i in d_k:
  dx.append(i[0])
  dy.append(i[1])

gradx=[]
grady=[]
for i in grad_k:
  gradx.append(i[0])
  grady.append(i[1])

def plot_function(f, title, dom = np.linspace(-5, 5, 500), angle = (30,40)):

  plt.style.use('fivethirtyeight')

  

  x, y = dom, dom
  X, Y = np.meshgrid(x, y)
  Z = f([X, Y])

  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(projection = '3d')
  ax.plot(np.array(p_k)[:,0],np.array(p_k)[:,1], linestyle='--', marker='o', color='black', linewidth = 3)
  ax.plot(np.array(p_k)[-1,0], np.array(p_k)[-1,1], 'ro', markersize = 11)
  ax.set_title(title)
  ax.set_xlabel('$x_1$')
  ax.set_ylabel('$x_2$')
  ax.set_zlabel('$f(x_1, x_2)$')
  ax.plot_surface(X, Y, Z, cmap='jet')
  ax.view_init(angle[0], angle[1])
  plt.tight_layout()
  plt.show()

def plot_results(f, dim=[np.linspace(-1, 7, 500), np.linspace(-1, 7, 500)]):
 
    plt.style.use('fivethirtyeight')

    title = 'Gradiente Projetado'

    
    x, y = dim[0], dim[1]
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    fig, (ax1) = plt.subplots(1, figsize=(15, 8))
    plt.suptitle(title, y=1.05)

    
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.set_axisbelow(False)

    
    x_grid, y_grid = np.mgrid[dim[0][0]:dim[0][-1]:100j, dim[1][0]:dim[1][-1]:100j]
    pontos = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    valores_restricoes = np.array([restricao(ponto) -vet_b()for ponto in pontos])
    pontos_facteis = np.all(valores_restricoes <= 0, axis=1)

    
    ax1.scatter(pontos[pontos_facteis, 0], pontos[pontos_facteis, 1], color='green', alpha=0.5, label='Região Factível')

    
    ax1.plot(np.array(p_k)[:, 0], np.array(p_k)[:, 1], linestyle='--', marker='o', color='black', linewidth=3)
    ax1.plot(np.array(p_k)[-1, 0], np.array(p_k)[-1, 1], 'ro', markersize=11, label='Último ponto')
    ax1.set(title='Caminho durante a otimização - Curvas de Nível', xlabel='x1', ylabel='x2')
    CS = ax1.contour(X, Y, Z, 10, cmap='jet')
    ax1.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax1.legend()


    plt.show()

def plot_f_obj(f_x):
   
    fig, ax2 = plt.subplots(figsize=(8, 8))
    # Valor da função custo em cada iteração
    ax2.plot(f_x, linestyle='--', marker='o', color='black')
    ax2.plot(len(f_x) - 1, f_x[-1], 'ro', markersize=11,label=f'Valor ótimo({f_x[-1]:.2f})')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(title='Valor da função custo durante a otimização', xlabel='Iterações', ylabel='Valor da função objetivo')
    ax2.legend()
    plt.show()

def plot_lambda(l):
   
    fig, ax2 = plt.subplots(figsize=(8, 8))
    # Valor da função custo em cada iteração
    ax2.plot(l, linestyle='--', marker='o', color='black')
    ax2.plot(len(l) - 1, l[-1], 'ro', markersize=11,label=f'Valor final lambda({l[-1]:.2f})')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(title='Valor de Lambda a cada iteração', xlabel='Iterações', ylabel='Valores de lambda')
    ax2.legend()
    plt.show()

plot_lambda(l)

"""#Plotando"""

plot_f_obj(f_x)

plot_results(funcao)

plot_function(funcao, title = 'F(x,y)')
