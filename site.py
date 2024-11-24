import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

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
        ax.imshow(self.state, cmap='RdYlGn', vmin=-1, vmax=1, interpolation='nearest')  # Reduz a suavização
        for (i, j) in self.influencers:
            ax.text(j, i, "★", ha='center', va='center', color="black", fontsize=8)
        for (i, j) in self.wise_people:
            ax.text(j, i, "♦", ha='center', va='center', color="black", fontsize=8)  # Símbolo de lampada para sábios
        ax.set_title(f"Iteração: {iteration} - {self.fake_news_name}")
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
    num_iterations = st.slider("Número de Iterações", min_value=1, max_value=1000, value=100, step=1)

    # Criar o botão de iniciar simulação
    start_button = st.button("Iniciar Simulação")

    if start_button:
        # Inicializar o modelo com os parâmetros escolhidos
        model = FakeNewsIsingModel(
            grid_size=grid_size,
            num_influencers=num_influencers,
            num_wise=num_wise,
            temperature=temperature,
            fake_news_name=fake_news_name
        )

        # Criar o gráfico da grade inicial com resolução menor
        col1, col2 = st.columns([1, 1])  # Criar duas colunas de largura igual
        with col1:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=40)  # Reduzimos figsize e dpi
            ax.set_axis_off()  # Oculta os eixos
            model.plot_grid(0, ax)
            st.pyplot(fig)

        # Barra de carregamento para iteração
        progress_bar = st.progress(0)

        # Loop de iteração
        for iteration in range(num_iterations):
            model.update_state()  # Atualiza o estado do modelo
            model.calculate_credibility()  # Calcula a credibilidade

            progress_bar.progress((iteration + 1) / num_iterations)

            # Intervalo de atualização para dar tempo ao gráfico
            time.sleep(0.1)

        # Exibir a tela final com o gráfico de credibilidade
        st.subheader(f"Credibilidade da Fake News ({fake_news_name}) ao Longo do Tempo")
        st.line_chart(model.credibility_history)  # Exibe o gráfico final de credibilidade

        with col2:
            # Exibir a grade final (após simulação) com resolução menor
            fig, ax = plt.subplots(figsize=(4, 4), dpi=40)  # Reduzimos figsize e dpi
            ax.set_axis_off()
            model.plot_grid(num_iterations, ax)
            st.pyplot(fig)

# Configuração da página e abas
st.set_page_config(page_title="Simulação de Fake News", layout="wide")

# Aba principal
tab1, tab2 = st.tabs(["Simulação", "Sobre"])

with tab1:
    run_simulation()
    
with tab2:
    st.header("Sobre o Modelo")
    st.markdown("""
    Este simulador utiliza uma versão modificada do **Modelo de Ising**, tradicionalmente utilizado em física para simular ferromagnetismo, agora adaptado para modelar a propagação de **fake news** em uma sociedade. Este modelo nos ajuda a entender como crenças podem se espalhar e como diferentes fatores, como influenciadores e especialistas (sábios), podem afetar essa propagação. Foi desenvolvido por estudantes do quarto semestre da Ilum Escola de Ciências.

    ### Importância de modelar fake news:
    Modelar a propagação de fake news é crucial para entender os mecanismos que levam à sua disseminação. Fake news podem influenciar escolhas políticas, causar pânico social e afetar decisões econômicas. Ao entender os padrões de propagação, podemos buscar formas de mitigar seus efeitos e desenvolver estratégias para aumentar a conscientização da população.

    ### Modificações do modelo:
    - **Spins (-1, 0, 1)** representam:
      - `1` (verde): Pessoas que acreditam na fake news.
      - `0`: Pessoas neutras, que não têm opinião formada sobre a fake news.
      - `-1` (vermelho): Pessoas que não acreditam na fake news.
    - **Influenciadores (★)** têm maior peso e raio de influência na atualização de vizinhos. 
    - **Sábios (♦)** nunca acreditam na fake news, representando aqueles que têm conhecimento especializado e estão imunes à influência de notícias falsas.

    ### Parâmetros ajustáveis:
    - **Temperatura**: Controla a probabilidade de mudanças de estado em uma pessoa. Quanto maior a temperatura, maior a chance de uma pessoa mudar de crença, mesmo contra sua vontade. Isso simula uma sociedade mais volúvel ou suscetível a influências externas.
    - **Influenciadores e sábios**: Afetam diretamente a dinâmica local do modelo. A presença de influenciadores pode acelerar a propagação da fake news, enquanto os sábios ajudam a frear essa disseminação.

    ### Fórmula de Atualização:
    A fórmula de atualização usada no modelo de Ising modificado segue a ideia básica da termodinâmica e da física estatística, onde cada indivíduo (ou spin) tenta minimizar sua energia local. A atualização de um estado envolve a interação com seus vizinhos e a probabilidade de mudança é dada por:
    """)
    
    st.latex(r'''
    P(\Delta E) = \frac{1}{1 + e^{\frac{\Delta E}{T}}}
    ''')

    st.markdown("""
    Onde:
    - $\Delta E$ é a variação de energia ao mudar o estado de um indivíduo.
    - $T$ é a temperatura do sistema (quanto maior a temperatura, mais fácil é a mudança de estado).
    - O valor de $\Delta E$ é calculado como a diferença entre a energia do estado atual e do novo estado proposto, levando em consideração a influência dos vizinhos e dos influenciadores.
    """)

    st.markdown("""
    ### Funcionamento do Algoritmo:
    - **Inicialização**: No começo, cada pessoa é atribuída a uma crença aleatória: ela pode acreditar na fake news (`1`), ser neutra (`0`) ou não acreditar nela (`-1`). 
    - **Seleção aleatória**: Durante cada iteração, uma pessoa é escolhida aleatoriamente para tentar mudar de opinião. Se essa pessoa for um sábio (que nunca acredita na fake news), ela não terá sua crença alterada. Caso contrário, ela poderá ser influenciada pelos seus vizinhos e, no caso dos **influenciadores**, por sua maior capacidade de persuasão.
    - **Probabilidade de mudança de crença**: Para uma pessoa que não seja sábia, a mudança de crença depende de sua interação com os vizinhos, o que é calculado pela variação de energia ($\Delta E$) de sua configuração. Caso essa mudança de energia seja negativa (ou com uma probabilidade calculada via a fórmula ($P(\Delta E$), ela pode mudar de crença.

    Ajustar os parâmetros é essencial para modelar a propagação de fake news de maneira mais personalizada. Por exemplo, os sábios podem ser vistos como especialistas. Sendo assim, assuntos médicos como vacinas terão mais sábios que assuntos obscuros, como "pinguins extraterrestres que invadiram o planeta há duas eras geológicas atrás".  
    Sinta-se à vontade para ajustar os parâmetros e observar os efeitos na propagação de crenças!
    """)
