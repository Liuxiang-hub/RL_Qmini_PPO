import torch.nn as nn
import torch


class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, rnn_layer, num_layer, input_size, num_hiddens, output_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        # # 定义RNN层
        # RNN输入的形状为（num_steps, batch_size, input_size）  # input_size 就是 vocab_size
        # RNN输出的形状为（num_steps, batch_size, num_hiddens）和（1, batch_size, num_hiddens）
        self.rnn = rnn_layer(input_size, num_hiddens, num_layer)
        self.input_size = self.rnn.input_size
        self.num_hiddens = self.rnn.hidden_size
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.01, std=0.01)
        self.output_size = output_size
        self.linear = nn.Linear(self.num_hiddens, self.output_size)

    def forward(self, inputs, state):
        # inputs的形状为（num_steps, num_hiddens, input_size）
        # Y是所有时间步的隐藏状态，state是最后一个时间步的隐藏状态
        # Y的形状为（num_steps, batch_size, num_hiddens），state为（1，batch_size, num_hiddens）
        Y, state = self.rnn(inputs, state)
        # 全连接层首先将Y的形状改为(num_steps*batch_size, num_hiddens)
        # 它的输出形状是(num_steps*batch_size, output_size)。
        output = self.linear(Y[-1].reshape((-1, self.num_hiddens)))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device))


class RNNNet(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNNet, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [b, seq, h]
        out = out.view(-1, self.hidden_size)
        out = self.linear(out)  # [seq,h] => [seq,3]
        out = out.unsqueeze(dim=0)  # => [1,seq,3]
        return out, hidden_prev

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_layers, batch_size, self.hidden_size),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_layers, batch_size, self.hidden_size),
                                device=device),
                    torch.zeros((self.num_layers, batch_size, self.hidden_size),
                                device=device))


class CNN_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size, batch_size, seq_length) -> None:
        super(CNN_LSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_directions = 1  # 单向LSTM
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=64, seq_len=3, input_size=3) ---> permute(0, 2, 1)
        # (64, 3, 3)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=2),  # shape(7,--)  ->(64,3,2)
            nn.ReLU())
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        batch_size, seq_len = x.size()[0], x.size()[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0))
        pred = self.fc(output)
        pred = pred[:, -1, :]
        return pred
