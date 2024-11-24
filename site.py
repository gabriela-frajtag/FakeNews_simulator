import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Modelo de Ising modificado para fake news
class FakeNewsIsingModel:
    def __init__(self, grid_size=20, steps_per_update=10, num_influencers=2, num_wise=3, temperature=2, fake_news_name="Fake News"):
        self.grid_size = grid_size
        self.steps_per_update = steps_per_update
        self.num_influencers = num_influencers
        self.num_wise = num_wise
        self.temperature = temperature
        self.fake_news_name = fake_news_name
        self.state = np.random.choice([-1, 0, 1], (grid_size, grid_size))
        self.influencers = self.generate_influencers()
        self.wise_people = self.generate_wise_people()
        self.credibility_history = []

    def generate_influencers(self):
        influencers = set()
        while len(influencers) < self.num_influencers:
            i, j = np.random.randint(0, self.grid_size, size=2)
            influencers.add((i, j))
        return influencers

    def generate_wise_people(self):
        wise_people = set()
        while len(wise_people) < self.num_wise:
            i, j = np.random.randint(0, self.grid_size, size=2)
            if (i, j) not in self.influencers:
                wise_people.add((i, j))
                self.state[i, j] = -1  # Não acredita na fake news
        return wise_people

    def get_neighbors(self, i, j, radius=1):
        neighbors = []
        for x in range(i - radius, i + radius + 1):
            for y in range(j - radius, j + radius + 1):
                if (x, y) != (i, j):
                    neighbors.append((x % self.grid_size, y % self.grid_size))
        return neighbors

    def update_state(self):
        for _ in range(self.steps_per_update):
            i, j = np.random.randint(0, self.grid_size, size=2)
            if (i, j) in self.wise_people:
                continue  # Ignora os sábios, pois eles não acreditam na fake news
            new_state = np.random.choice([-1, 0, 1])
            delta_e = self.energy_change(i, j, new_state)
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / self.temperature):
                self.state[i, j] = new_state

    def energy_change(self, i, j, new_state):
        radius = 2 if (i, j) in self.influencers else 1
        neighbors = self.get_neighbors(i, j, radius)
        interaction_weight = 2 if (i, j) in self.influencers else 1
        old_energy = -interaction_weight * sum(self.state[i, j] * self.state[x, y] for x, y in neighbors)
        new_energy = -interaction_weight * sum(new_state * self.state[x, y] for x, y in neighbors)
        return new_energy - old_energy

    def plot_grid(self, ax):
        ax.imshow(self.state, cmap='RdYlGn', vmin=-1, vmax=1)
        for (i, j) in self.influencers:
            ax.text(j, i, "★", ha='center', va='center', color="black", fontsize=10)
        for (i, j) in self.wise_people:
            ax.text(j, i, "💡", ha='center', va='center', color="black", fontsize=10)
        ax.set_title(f"Fake News: {self.fake_news_name}")
        ax.grid(False)
        ax.axis('off')

# Função para exibir a animação no Streamlit
def run_simulation():
    # Definir variáveis de controle
    grid_size = st.sidebar.slider("Tamanho do grid", 10, 50, 20)
    steps_per_update = st.sidebar.slider("Passos por atualização", 1, 50, 10)
    num_influencers = st.sidebar.slider("Número de influenciadores", 1, 10, 2)
    num_wise = st.sidebar.slider("Número de sábios", 1, 10, 3)
    temperature = st.sidebar.slider("Temperatura", 0.1, 5.0, 2.0)
    fake_news_name = st.sidebar.text_input("Nome da Fake News", "Fake News")

    # Instanciando o modelo
    model = FakeNewsIsingModel(
        grid_size=grid_size, 
        steps_per_update=steps_per_update,
        num_influencers=num_influencers, 
        num_wise=num_wise,
        temperature=temperature,
        fake_news_name=fake_news_name
    )

    # Criação da figura e eixo
    fig, ax = plt.subplots(figsize=(5, 5))

    # Função de animação
    def animate(frame):
        model.update_state()
        ax.clear()  # Limpa o gráfico anterior
        model.plot_grid(ax)

    # Configurar a animação
    anim = FuncAnimation(fig, animate, frames=100, interval=200, repeat=False)

    # Exibir a animação no Streamlit
    st.pyplot(fig)

# Exibir a simulação
if __name__ == "__main__":
    run_simulation()
