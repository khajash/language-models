import torch
from torch import nn, Tensor


class RecurrentModel(nn.Module):
    def __init__(self, ntoken, d_model, num_layers=2, dropout=0.1, cell_type="GRU") -> None:
        super().__init__()

        self.embedding = nn.Embedding(ntoken, d_model)
        self.rnn = getattr(nn, cell_type)(
            input_size=d_model, 
            hidden_size=d_model,
            num_layers=num_layers,
            bias=True,
            dropout=dropout,
            bidirectional=False
        )
        self.prediction = nn.Linear(d_model, ntoken)

    def forward(self, x: Tensor, hidden: Tensor):

        x = self.embedding(x)
        x, hidden_n = self.rnn(x, hidden)
        out = self.prediction(x)
        return out, hidden_n


if __name__ == "__main__":
    ntoken = 1000
    d_model = 100
    num_layers = 2
    batch_size, seq_len = 20, 35
    rnn = RecurrentModel(ntoken, d_model, num_layers, 0.1, "GRU")
    data = torch.randint(0, ntoken, (seq_len, batch_size), dtype=torch.long)
    hidden = torch.zeros(num_layers, batch_size, d_model)

    print(rnn)

    print("Finished initializing")
    print("input: ", data.shape)

    out, hidden_n = rnn(data, hidden)
    print(out.shape, hidden_n.shape)
