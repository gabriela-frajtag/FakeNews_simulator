import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time  # Importando a biblioteca time para usar time.sleep()

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
                continue  # Ignora os sábios
            new_state = np.random.choice([-1, 0, 1])
            delta_e = self.energy_change(i, j, new_state)
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / self.temperature):
                self.state[i, j] = new_state

    def calculate_credibility(self):
        credibility = np.mean(self.state)
        self.credibility_history.append(credibility)

    def plot_grid(self, iteration, ax):
        ax.imshow(self.state, cmap='RdYlGn', vmin=-1, vmax=1)
        for (i, j) in self.influencers:
            ax.text(j, i, "★", ha='center', va='center', color="black", fontsize=10)
        for (i, j) in self.wise_people:
            ax.text(j, i, "💡", ha='center', va='center', color="black", fontsize=10)  # Símbolo de lampada para sábios
        ax.set_title(f"Iteração: {iteration}")
        ax.grid(True, color="black", linewidth=0.5)
        ax.axis('off')

# Função principal para a interface do Streamlit
def run_simulation():
    st.title("Simulador de Propagação de Fake News - Modelo de Ising")

    # Inputs do usuário
    fake_news_name = st.text_input("Nome da Fake News", "Fake News")
    grid_size = st.slider("Tamanho da grade", min_value=10, max_value=50, value=20)
    num_influencers = st.slider("Número de Influenciadores", min_value=1, max_value=10, value=2)
    num_wise = st.slider("Número de Sábios", min_value=1, max_value=10, value=3)
    temperature = st.slider("Temperatura", min_value=0.1, max_value=5.0, value=2.0)

    # Inicializar o modelo com os parâmetros escolhidos
    model = FakeNewsIsingModel(
        grid_size=grid_size,
        num_influencers=num_influencers,
        num_wise=num_wise,
        temperature=temperature,
        fake_news_name=fake_news_name
    )

    # Criar o espaço para exibir o gráfico animado
    placeholder = st.empty()

    # Criar o gráfico
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()  # Oculta os eixos

    # Loop de iteração
    for iteration in range(100):
        model.update_state()  # Atualiza o estado do modelo
        model.calculate_credibility()  # Calcula a credibilidade
        ax.clear()  # Limpa o gráfico
        model.plot_grid(iteration, ax)  # Plota a nova grade
        st.pyplot(fig)  # Exibe o gráfico atualizado
        st.line_chart(model.credibility_history)  # Exibe a credibilidade ao longo do tempo
        time.sleep(0.1)  # Intervalo de atualização para dar tempo ao gráfico

if __name__ == "__main__":
    run_simulation()

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
    Foi desenvolvido por alunos do quarto semestre da Ilum Escola de Ciência.
    
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

    Ajustar os parâmetros é essencial para modelar a propagação de fake news de maneira mais personalizada. Por exemplo, os sábios podem ser vistos como espacialistas. Sendo assim, assuntos médicos como vacinas terão mais sábios que assuntos obscuros, como "pinguins extraterrestres que invadiram o planeta há duas eras geológicas atrás".  
    Sinta-se à vontade para ajustar os parâmetros e observar os efeitos na propagação de crenças!
    """)
