import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

class FakeNewsIsingModel:
    def __init__(self, grid_size=20, steps_per_update=10, num_influencers=2, num_wise=3, temperature=2):
        self.grid_size = grid_size
        self.steps_per_update = steps_per_update
        self.num_influencers = num_influencers
        self.num_wise = num_wise
        self.temperature = temperature
        self.state = np.random.choice([-1, 0, 1], (grid_size, grid_size))
        self.influencers = self.generate_influencers()
        self.wise_people = self.generate_wise_people()
        self.credibility_history = []

    def generate_influencers(self):
        influencers = set()
        while len(influencers) < self.num_influencers:
            i, j = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            influencers.add((i, j))
        return influencers

    def generate_wise_people(self):
        wise_people = set()
        while len(wise_people) < self.num_wise:
            i, j = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (i, j) not in self.influencers:
                wise_people.add((i, j))
                self.state[i, j] = -1  # Fixado para não acreditar
        return wise_people

    def get_neighbors(self, i, j, radius=1):
        neighbors = []
        for x in range(i - radius, i + radius + 1):
            for y in range(j - radius, j + radius + 1):
                if (x, y) != (i, j):
                    neighbors.append((x % self.grid_size, y % self.grid_size))
        return neighbors

    def energy_change(self, i, j, new_state):
        radius = 2 if (i, j) in self.influencers else 1
        neighbors = self.get_neighbors(i, j, radius)
        interaction_weight = 2 if (i, j) in self.influencers else 1
        old_energy = -interaction_weight * sum(self.state[i, j] * self.state[x, y] for x, y in neighbors)
        new_energy = -interaction_weight * sum(new_state * self.state[x, y] for x, y in neighbors)
        return new_energy - old_energy

    def update_state(self):
        for _ in range(self.steps_per_update):
            i, j = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (i, j) in self.wise_people:
                continue  # Ignora os sábios, pois eles não acreditam na fake news
            new_state = random.choice([-1, 0, 1])
            delta_e = self.energy_change(i, j, new_state)
            if delta_e < 0 or random.random() < np.exp(-delta_e / self.temperature):
                self.state[i, j] = new_state

    def calculate_credibility(self):
        credibility = np.mean(self.state)
        self.credibility_history.append(credibility)

    def plot_grid(self, iteration):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.state, cmap='RdYlGn', vmin=-1, vmax=1)
        for (i, j) in self.influencers:
            ax.text(j, i, "★", ha='center', va='center', color="black", fontsize=10)
        for (i, j) in self.wise_people:
            ax.text(j, i, "💡", ha='center', va='center', color="yellow", fontsize=10)
        ax.set_title(f"Iteração: {iteration}")
        ax.grid(True, color="black", linewidth=0.5)
        ax.axis('off')
        return fig

    def plot_credibility(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.credibility_history, label="Crença Média", color="blue")
        ax.set_title("Evolução da Crença Média")
        ax.set_xlabel("Iteração")
        ax.set_ylabel("Crença Média")
        ax.legend()
        ax.grid()
        return fig


# Streamlit App
st.title("Simulador de Propagação de Fake News - Modelo Ising")

# Configurações
grid_size = st.sidebar.slider("Tamanho do Grid", 10, 50, 20)
temperature = st.sidebar.slider("Temperatura", 0.1, 5.0, 2.0)
num_influencers = st.sidebar.slider("Número de Influenciadores", 1, 10, 2)
num_wise = st.sidebar.slider("Número de Sábios", 1, 10, 3)
steps_per_update = st.sidebar.slider("Passos por Atualização", 1, 50, 10)

# Iniciar o modelo
model = FakeNewsIsingModel(
    grid_size=grid_size,
    temperature=temperature,
    num_influencers=num_influencers,
    num_wise=num_wise,
    steps_per_update=steps_per_update,
)

# Botão para executar a simulação
if st.button("Iniciar Simulação"):
    for iteration in range(1, 21):  # Simulação de 20 iterações
        model.update_state()
        model.calculate_credibility()
        grid_fig = model.plot_grid(iteration)
        cred_fig = model.plot_credibility()

        # Exibir os gráficos
        st.pyplot(grid_fig)
        st.pyplot(cred_fig)
