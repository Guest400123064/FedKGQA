import torch.nn as nn
from easydict import EasyDict


class EncoderSeqRNN(nn.Module):

    def __init__(self, config: EasyDict):
        super(EncoderSeqRNN, self).__init__()
        
        self.config = config
        self.rnn = nn.RNN(
            config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )
        return

    def forward(self, x):

        return self.rnn(x)

    def embed_only(self, x):

        _, hid_state = self.forward(x)
        return hid_state[-1] 


class EncoderSeqLinear(nn.Module):
    
    def __init__(self, config: EasyDict):
        super(EncoderSeqLinear, self).__init__()
        
        self.config = config
        self.linear = nn.Linear(config.seq_len, 1)
        return

    def forward(self, x):

        b, s, e = x.shape
        x_t = x.view(b, e, s)
        return self.linear(x_t).view(b, -1)

    def embed_only(self, x):

        return self.forward(x)


class EncoderSeqLSTM(nn.Module):
    pass


class EncoderSeqGRU(nn.Module):
    pass


class EncoderSeqCNN(nn.Module):
    pass

 
class SequenceEncoder(nn.Module):

    _LIB_ENCODER = {
        "rnn": EncoderSeqRNN,
        "lin": EncoderSeqLinear,
        "gru": EncoderSeqGRU,
        "lstm": EncoderSeqLSTM,
        "cnn": EncoderSeqCNN
    }

    SAMPLE_CONFIG = EasyDict({
        "rnn": {
            "cls": "rnn",
            "input_size": 50,
            "hidden_size": 32,
            "num_layers": 3
        },
        "lin": {
            "cls": "lin",
            "seq_len": 128,
            "input_size": 50
        }
    })
    
    def __init__(self, config: EasyDict):
        super(SequenceEncoder, self).__init__()

        encoder_cls = SequenceEncoder._LIB_ENCODER.get(config.cls)
        self.encoder = encoder_cls(config)
        self.config = config
        return

    def forward(self, seq):

        return self.encoder(seq)

    def embed_only(self, seq):

        return self.encoder.embed_only(seq)
