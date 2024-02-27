import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', is_slippery=True) 

# Definir el espacio de estados y acciones
estados = env.observation_space.n
acciones = env.action_space.n

# Inicializar la matriz Q
Q = np.zeros((estados, acciones))

# Parámetros
tasa_aprendizaje = 0.1
gamma = 0.9
epsilon = 0.1

# Lista para almacenar las recompensas
recompensas = []

# Bucle de entrenamiento
for episodio in range(1000):
    # Obtener el estado inicial
    estado_actual = env.reset()

    # Episodio
    while True:
        # Seleccionar una acción según la política actual (epsilon-greedy)
        if np.random.rand() < epsilon:
            accion_actual = np.argmax(Q[int(estado_actual), :])

        else:
            accion_actual = np.argmax(Q[int(estado_actual), :])

        # Realizar la acción
        estado_siguiente, recompensa, hecho, _ = env.step(accion_actual)

        # Actualizar la matriz Q
        Q[estado_actual, accion_actual] = Q[estado_actual, accion_actual] + tasa_aprendizaje * (recompensa + gamma * np.max(Q[estado_siguiente, :]) - Q[estado_actual, accion_actual])

        # Actualizar el estado actual
        estado_actual = estado_siguiente

        # Salir del episodio si se ha llegado al final
        if hecho:
            break

    # Almacenar la recompensa del episodio
    recompensas.append(recompensa)

# Evaluación
# Jugar el juego con la política aprendida
recompensa_promedio = np.mean(recompensas[-100:])

# Mostrar la recompensa promedio
print("Recompensa promedio:", recompensa_promedio)

# Graficar las recompensas
plt.plot(recompensas)
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.show()
