# IA_ChatBot


🤖 Chatbot com Seq2Seq e Atenção em PyTorch

Implementação de um chatbot conversacional como exercício bônus da disciplina de Inteligência Artificial.
Este projeto utiliza uma arquitetura Encoder-Decoder com Mecanismo de Atenção desenvolvida em PyTorch, capaz de gerar respostas a partir de entradas textuais.
🧰 Tecnologias Utilizadas

    🐍 Python 3.10+

    🔥 PyTorch

    📓 Jupyter Notebook

    🗂️ Dataset: DailyDialog

🚀 Como Executar

    Clone o repositório

git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO

    Crie um ambiente virtual e instale as dependências

python -m venv venv
source venv/bin/activate     # macOS/Linux
# .\venv\Scripts\activate    # Windows

pip install -r requirements.txt

    O arquivo requirements.txt deve incluir:
    torch, datasets, scikit-learn, entre outras dependências necessárias.

    Execute o chatbot

        Use o script de inferência ou as células do notebook para iniciar uma conversa.

        Certifique-se de que os arquivos best_chatbot_model.pth e vocabulary.pkl estejam na pasta raiz do projeto.

📌 Estado do Projeto

🧪 Prova de Conceito
Este modelo ainda sofre com overfitting devido a limitações de tempo e recursos computacionais no treinamento.
⚠️ As respostas podem não ser coerentes ou realistas, e o projeto tem como objetivo principal demonstrar a funcionalidade da arquitetura Seq2Seq com Atenção, além dos desafios práticos de treinar modelos de linguagem.
📬 Contato

Caso tenha interesse em colaborar, sugestões ou dúvidas, fique à vontade para abrir uma issue ou pull request.
