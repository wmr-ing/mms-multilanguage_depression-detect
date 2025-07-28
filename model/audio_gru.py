import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.signal.windows import lanczos
from transformers import AutoModel

class AudioBiLSTM(nn.Module):
    def __init__(self, config):
        super(AudioBiLSTM, self).__init__()
        self.num_classes = config['num_classes']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']
        self.num_classes1 = config['num_classes1']
        self.build_model()
        # self.init_weight()

    def init_weight(net):
        for name, param in net.named_parameters():
            if not 'ln' in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def build_model(self):
        # attention layer
        self.attention_layer1 = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True))
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        #self.lstm_net_audio = nn.LSTM(self.embedding_size,
        #                         self.hidden_dims,
        #                         num_layers=self.rnn_layers,
        #                         dropout=self.dropout,
        #                         bidirectional=self.bidirectional,
        #                         batch_first=True)
        self.lstm_net_audio = nn.GRU(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout, batch_first=True)

        self.audio_ln = nn.LayerNorm(self.embedding_size)

        # modal attention
        self.attention_query2 = nn.Linear(self.hidden_dims,
                                         self.hidden_dims,
                                         bias=False)
        self.attention_key2 = nn.Linear(self.hidden_dims,
                                       self.hidden_dims, bias=False)
        self.attention_value2 = nn.Linear(self.hidden_dims,
                                         self.hidden_dims,
                                         bias=False)
        # 最终融合的线性变换
        self.fusion_layer2 = nn.Linear(self.hidden_dims,
                                      self.hidden_dims, bias=True)


        #语言全连接层
        self.fc_language2 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes1),
            # nn.ReLU(),
            nn.Softmax(dim=1)
        )
        # FC层
        self.fc_audio = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims,self.num_classes),
            nn.Softmax(dim=1)
            # nn.ReLU(),
            #nn.Softmax(dim=1)
        )
        self.audio_bn1 = nn.BatchNorm1d(self.embedding_size)
        self.audio_ln1 = nn.LayerNorm(self.hidden_dims)
        self.audio_ln2 = nn.LayerNorm(self.hidden_dims)
    def self_attention2(self, x):
        # 可学习参数，用于注意力计算
        # 计算 Query、Key 和 Value
        query = self.attention_query2(x)  # [batch_size, total_hidden_dims]
        key = self.attention_key2(x)  # [batch_size, total_hidden_dims]
        value = self.attention_value2(x)  # [batch_size, total_hidden_dims]
        # 计算注意力分数
        # 这里通过点积计算注意力权重，并使用 softmax 归一化
        d_k = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(d_k)  # [batch_size, batch_size]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, batch_size]
        # 加权求和
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, total_hidden_dims]
        # 融合结果（线性变换）
        fused_feature = self.fusion_layer2(attention_output)  # [batch_size, total_hidden_dims]
        return  fused_feature
    def attention_net_with_w1(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        #         h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
       # print(atten_w.shape, m.transpose(1, 2).shape)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    '''
    def forward(self, x,mask):
        #x = self.audio_ln(x)
        #x = self.audio_bn1(x)
        #lengths = mask.sum(dim=1).long().cpu()  # 将 lengths 移动到 CPU
        #x_pack = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        #x, _ = self.lstm_net_audio(x_pack)
        #x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x, _ = self.lstm_net_audio(x)
        mask = mask.unsqueeze(-1).float()
        x = x * mask
        mask_sum = mask.sum(dim=1)
        x_sum = x.sum(dim=1)
        x = x_sum / (mask_sum + 1e-6)  # 这里加一个小的常数防止除以零
        #x = self.audio_ln1(x)
        #x = self.audio_ln2(x)
        #x = self.self_attention2(x)
        #x = x.mean(dim=1)
        out = self.fc_audio(x)
        return out
    '''
    def forward(self, x, mask):
        #x = self.audio_ln(x)
        lengths = mask.sum(dim=1).long().cpu()  # 计算真实长度，移动到 CPU
        # 使用 pack_padded_sequence 处理填充部分
        x_pack = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # 通过 GRU 网络
        x_pack, _ = self.lstm_net_audio(x_pack)
        # 解包序列
        x_audio, _ = nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        #mask = mask.unsqueeze(-1).float()
        #x_audio = x_audio * mask
        #mask_sum = mask.sum(dim=1)
        #x_sum = x_audio.sum(dim=1)
        #x_audio = x_sum / (mask_sum + 1e-6)  # 这里加一个小的常数防止除以零
        x_sum = x_audio.sum(dim=1)
        x_audio = x_sum / (lengths.to(x_sum.device).float().unsqueeze(1) + 1e-6)
        x_audio = self.fc_audio(x_audio)
        return x_audio

'''
from mamba_ssm import Mamba

class AudioMamba(nn.Module):
    def __init__(self, config):
        super(AudioMamba, self).__init__()
        self.num_classes = config['num_classes']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.embedding_size = config['embedding_size']
        self.num_classes1 = config['num_classes1']

        # 替换 GRU 为 Mamba
        self.mamba = Mamba(
            d_model=self.embedding_size,  # 输入维度
            d_state=16,  # SSM 状态维度
            d_conv=4,  # 卷积核大小
            expand=2,  # 扩展因子
        )

        # 保持其他层不变
        self.audio_ln = nn.LayerNorm(self.embedding_size)
        self.fc_audio = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, mask):
        # 1. 输入预处理
        x = self.audio_ln(x)  # [batch, seq_len, embed_dim]

        # 2. Mamba 处理（自动处理可变长度）
        x = self.mamba(x)  # [batch, seq_len, embed_dim]

        # 3. 掩码处理（可选）
        if mask is not None:
            lengths = mask.sum(dim=1).long()
            x_sum = x.sum(dim=1)
            x = x_sum / (lengths.unsqueeze(1).float() + 1e-6)
        else:
            x = x.mean(dim=1)  # 全局平均池化

        # 4. 分类头
        return self.fc_audio(x)

'''
class AudioBiLSTM2(nn.Module):
    def __init__(self, config):
        super(AudioBiLSTM2, self).__init__()
        self.num_classes = config['num_classes']
        self.num_classes1 = config['num_classes1']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']

        self.build_model()
        # self.init_weight()

    def init_weight(net):
        for name, param in net.named_parameters():
            if not 'ln' in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def build_model(self):
        # attention layer
        self.attention_layer1 = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True))
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # self.lstm_net_audio = nn.LSTM(self.embedding_size,
        #                         self.hidden_dims,
        #                         num_layers=self.rnn_layers,
        #                         dropout=self.dropout,
        #                         bidirectional=self.bidirectional,
        #                         batch_first=True)
        self.lstm_net_audio = nn.GRU(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout, batch_first=True)

        self.audio_ln = nn.LayerNorm(self.embedding_size)

        # modal attention
        self.attention_query = nn.Linear(self.hidden_dims,
                                         self.hidden_dims,
                                         bias=False)
        self.attention_key = nn.Linear(self.hidden_dims,
                                       self.hidden_dims, bias=False)
        self.attention_value = nn.Linear(self.hidden_dims,
                                         self.hidden_dims,
                                         bias=False)
        # 最终融合的线性变换
        self.fusion_layer = nn.Linear(self.hidden_dims,
                                      self.hidden_dims, bias=True)
        #语言全连接层
        self.fc_language2 = nn.Sequential(
            #nn.Dropout(self.dropout),
            #nn.Linear(self.hidden_dims, self.hidden_dims),
            #nn.ReLU(),
            #nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes1),
            # nn.ReLU(),
            nn.Softmax(dim=1)
        )
        # FC层
        self.fc_audio = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1)
        )
    def self_attention(self, x):
        # 可学习参数，用于注意力计算
        # 计算 Query、Key 和 Value
        query = self.attention_query(x)  # [batch_size, total_hidden_dims]
        key = self.attention_key(x)  # [batch_size, total_hidden_dims]
        value = self.attention_value(x)  # [batch_size, total_hidden_dims]
        # 计算注意力分数
        # 这里通过点积计算注意力权重，并使用 softmax 归一化
        d_k = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(d_k)  # [batch_size, batch_size]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, batch_size]
        # 加权求和
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, total_hidden_dims]
        # 融合结果（线性变换）
        fused_feature = self.fusion_layer(attention_output)  # [batch_size, total_hidden_dims]
        return  fused_feature
    def attention_net_with_w1(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        #         h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
       # print(atten_w.shape, m.transpose(1, 2).shape)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x,mask):
        x = self.audio_ln(x)
        lengths = mask.sum(dim=1).long().cpu()  # 将 lengths 移动到 CPU
        x_pack = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm_net_audio(x_pack)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        mask = mask.unsqueeze(-1)
        x = x * mask
        mask_sum = mask.sum(dim=1)
        x_sum = x.sum(dim=1)
        x = x_sum / (mask_sum + 1e-6)  # 这里加一个小的常数防止除以
        language_out = self.fc_language2(x)
        #x = self.self_attention(x)
        #x = x.mean(dim=1)
        out = self.fc_audio(x)
        return out, language_out





