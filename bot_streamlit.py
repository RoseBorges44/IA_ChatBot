"""Streamlit interface for the DailyDialog chatbot

Coloque este arquivo na mesma pasta onde estão:
  • best_chatbot_model.pth
  • vocabulary.pkl

Execute:  streamlit run chatbot_streamlit.py
"""

import re
from pathlib import Path
import pickle

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# Utilidades de pré‑processamento
# -------------------------------------------------

def preprocess_text(text: str) -> str:
    """Lower + remove pontuação e espaços extras"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_to_indexes(sentence: str, vocabulary) -> list[int]:
    indexes = []
    for word in sentence.split():
        if word in vocabulary.word2index:
            indexes.append(vocabulary.word2index[word])
        else:
            indexes.append(vocabulary.word2index["<UNK>"])
    return indexes

# -------------------------------------------------
# Arquitetura – mesma da fase de treino (somente inferência)
# -------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            n_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, input_lengths):
        embedded = self.dropout(self.embedding(input_seq))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]

        # Combina direções também em hidden / cell
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_size).sum(1)
        cell = cell.view(self.n_layers, 2, -1, self.hidden_size).sum(1)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attn_weights = self.v(energy).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_size,
            hidden_size,
            n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = Attention(hidden_size)
        self.out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input_token))
        context, _ = self.attention(hidden[-1], encoder_outputs)
        lstm_in = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        lstm_out, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))
        output = torch.cat([lstm_out.squeeze(1), context], dim=1)
        output = self.out(output)
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def respond(self, sentence: str, vocabulary, max_length: int = 20) -> str:
        self.eval()
        with torch.no_grad():
            sentence = preprocess_text(sentence)
            idxs = sentence_to_indexes(sentence, vocabulary)
            idxs = [vocabulary.word2index["<SOS>"]] + idxs + [vocabulary.word2index["<EOS>"]]
            src = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(self.device)
            src_len = torch.tensor([len(idxs)], dtype=torch.long).to(self.device)

            enc_out, hidden, cell = self.encoder(src, src_len)
            input_token = torch.tensor([[vocabulary.word2index["<SOS>"]]], device=self.device)
            response_idxs = []
            for _ in range(max_length):
                output, hidden, cell = self.decoder(input_token, hidden, cell, enc_out)
                top1 = output.argmax(1).item()
                if top1 == vocabulary.word2index["<EOS>"]:
                    break
                response_idxs.append(top1)
                input_token = torch.tensor([[top1]], device=self.device)
            words = [vocabulary.index2word.get(i, "<UNK>") for i in response_idxs]
            return " ".join(words)

# -------------------------------------------------
# Cache de carregamento de modelo e vocabulário
# -------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="🔄 Carregando modelo e vocabulário…")
def load_chatbot():
    with open("vocabulary.pkl", "rb") as f:
        vocab = pickle.load(f)
    encoder = Encoder(vocab.n_words, 512, 256, 2, 0.1)
    decoder = Decoder(vocab.n_words, 512, 256, 2, 0.1)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    ckpt = torch.load("best_chatbot_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab

model, vocab = load_chatbot()

# -------------------------------------------------
# UI – Chat estilizado
# -------------------------------------------------

st.set_page_config(page_title="Chatbot DailyDialog", page_icon="🤖", layout="centered")

st.markdown("""
# 🤖 Chatbot DailyDialog
Converse comigo à vontade!
""")

if "history" not in st.session_state:
    st.session_state.history = []  # list[(speaker, text)]

# Exibe histórico
for speaker, text in st.session_state.history:
    avatar = "🗣️" if speaker == "user" else "🤖"
    with st.chat_message("user" if speaker == "user" else "assistant"):
        st.markdown(f"{text}")

# Entrada do usuário
user_msg = st.chat_input("Digite sua mensagem…")
if user_msg:
    # Append user message
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)
    # Bot responde
    response = model.respond(user_msg, vocab)
    st.session_state.history.append(("bot", response))
    with st.chat_message("assistant"):
        st.markdown(response)
