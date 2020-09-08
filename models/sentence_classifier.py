import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from .pretrained_models import PretrainedModel

class SentenceClassifier(nn.Module):
    def __init__(self, word_embed_size, hidden_size, vocab_size, num_classes,
            dropout_prob=0.5):
        super(SentenceClassifier, self).__init__()

        self.word_embed = nn.Embedding(vocab_size, word_embed_size, padding_idx=0)

        lstm1_input_size = word_embed_size
        self.hidden_size = hidden_size*2

        #  lstm
        #  self.lstm = nn.LSTM(lstm1_input_size, hidden_size, batch_first=True)
        #  self.linear = nn.Linear(hidden_size, num_classes)
        #  bilstm 1
        self.lstm = nn.LSTM(lstm1_input_size, hidden_size*2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, num_classes)
        #  bilstm 2 
        #  self.lstm = nn.LSTM(lstm1_input_size, hidden_size, batch_first=True, bidirectional=True)
        #  self.linear = nn.Linear(hidden_size*2, num_classes)
        
        self.init_weights()

        self.input_size = vocab_size
        self.output_size = num_classes
        self.dropout_prob = dropout_prob

        #  attenion without energy
        self.hi = nn.Linear(self.hidden_size*2, 1, bias=True)

        #  attenion with energy
        #  self.hi = nn.Linear(self.hidden_size*2, self.hidden_size*2, bias=True)
        #  self.attn_weights = nn.Linear(self.hidden_size*2, 1, bias=True)
        #  self.energy = nn.Linear(1, 1, bias=True)


    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def state_dict(self, *args, full_dict=False, **kwargs):
        return super().state_dict(*args, **kwargs)

    def forward(self, captions, lengths):
        embeddings = self.word_embed(captions)
        embeddings = F.dropout(embeddings, p=self.dropout_prob, training=self.training)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)
        
        batch_size, out_len, out_size = hiddens.shape
        
        # forward + backward
        hidden_out = hiddens.view(batch_size, out_len, 2, self.hidden_size)
        hidden_out = hidden_out[:, :, 0, :] + hidden_out[:, :, 1, :]
        hiddens = hidden_out
        
        #  sum
        #  hiddens = torch.sum(hidden_out, dim=1)
        #  lengths = torch.Tensor(lengths).cuda()
        #  last_hiddens = hiddens / torch.Tensor(lengths).cuda().type(hiddens.dtype).unsqueeze(1).expand(hiddens.shape)

        #  attention
        #  hidden_hi = self.hi.cuda()
        #  hidden_out = hidden_hi(hiddens)
        #  hidden_out = torch.tanh(hidden_out)
        
        #  mask
        #  mask = torch.arange(out_len)[None, :] < torch.LongTensor(lengths)[:, None].cpu()
        #  mask = mask.float()
        #  mask[mask==0] = -10000000
        #  mask = torch.unsqueeze(mask, 2)
        #  hidden_out = hidden_out * mask.cuda()  # hidden_out[128, 67, 4000]

        #  energy = hidden_out.squeeze(2)
        
        #  energy
        #  attn_weights = self.attn_weights.cuda()
        #  attn_weights = attn_weights(hidden_out)
        #  attn_weights = attn_weights * mask.cuda()

        #  attn_weights = torch.tanh(attn_weights)
        #  energy = self.energy.cuda()
        #  energy = energy(attn_weights) 
        #  energy = energy.squeeze(2) 

        #  mask
        #  mask = torch.arange(out_len)[None, :] < torch.LongTensor(lengths)[:, None].cpu()
        #  mask = mask.float()
        #  mask[mask==0] = -10000000
        #  energy = energy * mask.cuda()  # energy[128,120]

        #  attn
        #  soft_attn = nn.Softmax(energy, dim=1)
        #  soft_attn = torch.sigmoid(energy)
        #  soft_attn_sum = torch.sum(soft_attn, dim=1).reshape(-1,1)
        #  soft_attn = torch.div(soft_attn, soft_attn_sum)

        #  attn = torch.bmm(hidden_out.transpose(1,2), soft_attn.unsqueeze(2)).squeeze(2) # BxN 
        #  attn = torch.bmm(hiddens.transpose(1,2), soft_attn.unsqueeze(2)).squeeze(2) # BxN 
        #  last_hiddens = attn


        # Extract the outputs for the last timestep of each example
        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), hiddens.size(2))
        idx = idx.unsqueeze(1)
        idx = idx.to(hiddens.device)

        # Shape: (batch_size, hidden_size)
        last_hiddens = hiddens.gather(1, idx).squeeze(1)

        last_hiddens = F.dropout(last_hiddens, p=self.dropout_prob, training=self.training)
        outputs = self.linear(last_hiddens)
        return outputs
