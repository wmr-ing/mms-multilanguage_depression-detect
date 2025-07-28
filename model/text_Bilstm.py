import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextBiLSTM(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.num_classes = config['num_classes']
        self.num_classes1 = config['num_classes1']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']
        self.build_model()
        self.init_weight()

    def init_weight(net):
        for name, param in net.named_parameters():
            if 'ln' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def build_model(self):
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=self.bidirectional)

        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1),

        )

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
        self.ln1 = nn.LayerNorm(self.embedding_size)
        self.ln2 = nn.LayerNorm(self.hidden_dims)
    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result
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
    def forward(self, x, mask):
        x = x.permute(1, 0, 2)
        x = self.ln1(x)
        seq_lengths = mask.sum(dim=1).cpu()  # 计算每个样本的真实长度
        # 使用 pack_padded_sequence 处理填充部分
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        packed_output, (final_hidden_state, _) = self.lstm_net(packed_x)
        # 解包序列
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        output = output.permute(1, 0, 2)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        atten_out = self.ln2(atten_out)
        output = self.fc_out(atten_out)
        return output




class TextBiLSTM1(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM1, self).__init__()
        self.num_classes = config['num_classes']
        self.num_classes1 = config['num_classes1']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']
        self.build_model()
        self.init_weight()

    def init_weight(net):
        for name, param in net.named_parameters():
            if 'ln' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def build_model(self):
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=self.bidirectional)

        # FC层_语言预测
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_language = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes1),
            # nn.ReLU(),
            nn.Softmax(dim=1),
        )

        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1),
        )

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

        self.ln1 = nn.LayerNorm(self.embedding_size)
        self.ln2 = nn.LayerNorm(self.hidden_dims)
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

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result
    def forward(self, x,mask):
        x = x.permute(1, 0, 2)
        x = self.ln1(x)
        # x = self.input_layer1(x)
        # src_key_padding_mask = ~mask.bool()  # 转换为布尔掩码，False 表示填充部分
        # x = self.transformer(x, x, src_key_padding_mask=src_key_padding_mask)
        # x = self.output_layer1(x)
        output, (final_hidden_state, _) = self.lstm_net(x)
        # output : [batch_size, len_seq, n_hidden * 2]
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        mask = mask.unsqueeze(-1)
        output = output * mask
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        language_output = self.fc_language(atten_out)
        #atten_out = self.self_attention(atten_out)
        #atten_out = self.ln2(atten_out)
        output = self.fc_out(atten_out)
        return output, language_output

class TextBiLSTM3(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM3, self).__init__()
        self.num_classes = config['num_classes']
        self.num_classes1 = config['num_classes1']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']
        self.num_heads = 8
        self.encoder_layers = 3
        self.embedding_size1 = 512
        self.dropout1 = 0.3
        self.build_model()

    def init_weight(net):
        for name, param in net.named_parameters():
            if 'ln' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def build_model(self):
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )

        self.input_layer1 = nn.Linear(self.embedding_size, self.embedding_size1)
        self.transformer = nn.Transformer(
            d_model=self.embedding_size1,
            nhead=self.num_heads,
            num_encoder_layers=self.encoder_layers,
            dim_feedforward=self.embedding_size1 * 4,
            dropout=self.dropout1,
        )
        self.output_layer1 = nn.Linear(self.embedding_size1, self.embedding_size)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=self.bidirectional)

        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1),
        )

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

        self.ln1 = nn.LayerNorm(self.embedding_size)
        self.ln2 = nn.LayerNorm(self.hidden_dims)
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

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result
    def forward(self, x,mask):
        # x : [len_seq, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)
        x = self.ln1(x)
        #x = self.input_layer1(x)
        #src_key_padding_mask = ~mask.bool()  # 转换为布尔掩码，False 表示填充部分
        #x = self.transformer(x, x, src_key_padding_mask=src_key_padding_mask)
        #x = self.output_layer1(x)
        output, (final_hidden_state, _) = self.lstm_net(x)
        # output : [batch_size, len_seq, n_hidden * 2]
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        mask = mask.unsqueeze(-1)
        output = output * mask
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        #atten_out = self.ln2(atten_out)
        atten_out = self.self_attention(atten_out)
        #atten_out = self.ln2(atten_out)
        output = self.fc_out(atten_out)
        return output