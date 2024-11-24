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

    def energy_change(self, i, j, new_state):
        radius = 2 if (i, j) in self.influencers else 1
        neighbors = self.get_neighbors(i, j, radius)
        interaction_weight = 2 if (i, j) in self.influencers else 1
        old_energy = -interaction_weight * sum(self.state[i, j] * self.state[x, y] for x, y in neighbors)
        new_energy = -interaction_weight * sum(new_state * self.state[x, y] for x, y in neighbors)
        return new_energy - old_energy

    def update_state(self):
        for _ in range(self.steps_per_update):
            i, j = np.random.randint(0, self.grid_size, size=2)
            if (i, j) in self.wise_people:
                continue
            new_state = np.random.choice([-1, 0, 1])
            delta_e = self.energy_change(i, j, new_state)
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / self.temperature):
                self.state[i, j] = new_state

    def calculate_credibility(self):
        return np.mean(self.state)

# Função para animar o gráfico
def animate(i, model, ax):
    model.update_state()
    ax.clear()
    ax.imshow(model.state, cmap='RdYlGn', vmin=-1, vmax=1)
    for (x, y) in model.influencers:
        ax.text(y, x, "★", color="black", ha="center", va="center", fontsize=12)
    for (x, y) in model.wise_people:
        ax.text(y, x, "💡", color="yellow", ha="center", va="center", fontsize=12)
    ax.set_title(f"Iteração {i}")
    ax.axis("off")

# Configuração da interface com Streamlit
st.set_page_config(page_title="Simulação de Fake News", layout="wide")

# Aba principal
tab1, tab2 = st.tabs(["Simulação", "Sobre"])

with tab1:
    st.header("Simulação de Fake News com o Modelo de Ising")

    # Entrada de parâmetros
    grid_size = st.slider("Tamanho da grade (NxN):", 10, 50, 20)
    temperature = st.slider("Temperatura:", 0.1, 5.0, 2.0)
    num_influencers = st.slider("Quantidade de influenciadores:", 1, 10, 2)
    num_wise = st.slider("Quantidade de sábios:", 1, 10, 3)
    fake_news_name = st.text_input("Nome da Fake News:", "Fake News")

    # Inicializar o modelo
    model = FakeNewsIsingModel(grid_size, 10, num_influencers, num_wise, temperature, fake_news_name)

    # Exibir a animação
    st.write("### Simulação em andamento:")
    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, animate, fargs=(model, ax), frames=200, interval=200)
    st.pyplot(fig)

with tab2:
    st.header("Sobre o Modelo")
    st.markdown("""
    Este simulador utiliza uma versão modificada do **Modelo de Ising**, tradicionalmente utilizado em física para simular ferromagnetismo.
    
    ### Modificações do modelo:
    - **Spins (-1, 0, 1)** representam:
      - `-1`: Pessoas que acreditam na fake news.
      - `0`: Pessoas neutras.
      - `1`: Pessoas que não acreditam.
    - **Influenciadores (★)** têm maior peso na influência de vizinhos.
    - **Sábios (💡)** nunca acreditam na fake news.
    
    ### Parâmetros ajustáveis:
    - **Temperatura**: Controla a probabilidade de mudanças de estado.
    - **Influenciadores e sábios**: Afetam a dinâmica local do modelo.
    
    Sinta-se à vontade para ajustar os parâmetros e observar os efeitos na propagação de crenças!
    """)
