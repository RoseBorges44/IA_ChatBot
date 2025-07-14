# IA_ChatBot

Chatbot com Seq2Seq e Atenção em PyTorch

Implementação de um chatbot conversacional para o exercício bônus da disciplina de Inteligência Artificial. O modelo utiliza uma arquitetura Encoder-Decoder com Atenção em PyTorch para gerar respostas a partir de um texto de entrada.

Tecnologias Utilizadas

    Python 3.10+

    PyTorch

    Jupyter Notebook

    Dataset: DailyDialog 

Como Executar

    Clone o repositório:
    Bash

git clone [URL_DO_SEU_REPOSITORIO]
cd [NOME_DA_PASTA]

Crie o ambiente virtual e instale as dependências:
Bash

python -m venv venv
source venv/bin/activate  # No macOS/Linux
# .\venv\Scripts\activate  # No Windows
pip install -r requirements.txt

(O arquivo requirements.txt deve conter torch, datasets, scikit-learn, etc.)

Inicie o chat:
Para conversar com o modelo pré-treinado, execute o script de inferência ou a célula correspondente no notebook. Certifique-se que os arquivos 

    best_chatbot_model.pth e vocabulary.pkl estão na pasta.

Estado do Projeto

Prova de Conceito. O modelo atual sofre de overfitting significativo devido a limitações computacionais durante o treinamento. Como resultado, a qualidade das respostas é baixa e o projeto serve principalmente como uma demonstração funcional da arquitetura e dos desafios práticos do treinamento de modelos de linguagem.
