import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.dtensor.python.config import num_clients
from torch.autograd import Variable
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
from utils import ReverseLayerF

class Fusion_MMS(nn.Module):
    def __init__(self, config):
        super(Fusion_MMS, self).__init__()
        self.text_embed_size = config['text_embed_size']
        self.audio_embed_size = config['audio_embed_size']
        self.text_hidden_dims = config['text_hidden_dims']
        self.audio_hidden_dims = config['audio_hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        self.hidden_dims = config['hidden_dims']
        self.hidden_size = config['hidden_dims']
        self.num_classes = config['num_classes']
        self.output_size = config['num_classes']
        self.use_cmd_sim = config['use_cmd_sim']
        self.rnncell = config['rnncell']
        self.tanh = nn.Tanh()

        self.activation = self._get_activation(config['activation'])

        # ============================= TextBiLSTM =================================
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.text_hidden_dims, self.text_hidden_dims),
            nn.ReLU(inplace=True)
        )

        # 双层lstm
        self.lstm_net = nn.LSTM(self.text_embed_size, self.text_hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=True)

        # modal attention
        self.attention_query = nn.Linear(self.text_hidden_dims,
                                         self.text_hidden_dims,
                                         bias=False)
        self.attention_key = nn.Linear(self.text_hidden_dims,
                                       self.text_hidden_dims, bias=False)
        self.attention_value = nn.Linear(self.text_hidden_dims,
                                         self.text_hidden_dims,
                                         bias=False)

        self.fusion_layer = nn.Linear(self.text_hidden_dims,
                                      self.text_hidden_dims, bias=True)

        # FC层
        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.text_hidden_dims, self.text_hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.ln1 = nn.LayerNorm(self.text_embed_size)
        self.ln2 = nn.LayerNorm(self.text_hidden_dims)

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t',
                                  nn.Linear(in_features=self.text_hidden_dims, out_features=self.text_hidden_dims))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(self.hidden_dims))

        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1',
                                  nn.Linear(in_features=self.text_hidden_dims, out_features=self.hidden_dims))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())

        # ============================= TextBiLSTM =================================

        # ============================= AudioBiLSTM =============================

        self.lstm_net_audio = nn.GRU(self.audio_embed_size,
                                     self.audio_hidden_dims,
                                     num_layers=self.rnn_layers,
                                     dropout=self.dropout,
                                     bidirectional=True,
                                     batch_first=True)

        self.fc_audio = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.audio_hidden_dims, self.audio_hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=self.audio_hidden_dims, out_features=self.hidden_dims))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(self.hidden_dims))

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3',
                                  nn.Linear(in_features=self.audio_hidden_dims, out_features=self.hidden_dims))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        self.audio_ln = nn.LayerNorm(self.hidden_dims * 2)

        self.audio_ln = nn.LayerNorm(self.audio_embed_size)
        # ============================= AudioBiLSTM =============================
        # shared encoder
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1',
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(self.dropout))
            self.discriminator.add_module('discriminator_layer_2',
                                          nn.Linear(in_features=self.hidden_dims, out_features=len(self.hidden_dims)))
        ##########################################
        # shared-private collaborative discriminator
        ##########################################
        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=self.hidden_dims, out_features=4))
        # ============================= last fc layer =============================
        self.bn = nn.BatchNorm1d(self.text_hidden_dims + self.audio_hidden_dims)
        # modal attention
        self.modal_attn = nn.Linear(self.text_hidden_dims + self.audio_hidden_dims,
                                    self.text_hidden_dims + self.audio_hidden_dims, bias=False)
        self.fc_final = nn.Sequential(
            nn.Linear(self.text_hidden_dims + self.audio_hidden_dims,1),
            # nn.ReLU(),
            #nn.Softmax(dim=1),
            #nn.Sigmoid()
        )
        self.layers = nn.Sequential(
            nn.Linear(self.text_hidden_dims + self.audio_hidden_dims, 256),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, 1),
            #nn.Softmax(dim=1),
            # nn.Sigmoid()
        )

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.hidden_size * 6,
                                                           out_features=self.hidden_size * 3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=self.hidden_size * 3, out_features=2))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dims, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj_t = nn.Linear(self.text_hidden_dims, self.hidden_dims)
        self.proj_a = nn.Linear(self.audio_hidden_dims, self.hidden_dims)


        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dims))
        self.transformer = Transformer(
            d_model=self.hidden_dims,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=(self.hidden_dims) * 4,
            batch_first=True
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dims))

        self.fc = nn.Linear(self.hidden_dims, self.num_classes)

        self._initialize_weights()
    def _get_activation(self, name):
        return {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }.get(name.lower(), nn.ReLU())  # 默认返回ReLU
    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):

                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
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
    def reconstruct(self, ):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t,utterance_a):

        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_a = self.shared(utterance_a)

    def pretrained_feature(self, text_feature, audio_feature, mask):
        with torch.no_grad():
            x_text = text_feature.permute(1, 0, 2)
            x_text = self.ln1(x_text)
            seq_lengths = mask.sum(dim=1).cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(x_text, seq_lengths, enforce_sorted=False)
            packed_output, (final_hidden_state, _) = self.lstm_net(packed_x)
            # 解包序列
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
            output = output.permute(1, 0, 2)
            # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
            final_hidden_state = final_hidden_state.permute(1, 0, 2)
            atten_out = self.attention_net_with_w(output, final_hidden_state)
            atten_out = self.ln2(atten_out)
            text_feature1 = self.fc_out(atten_out)
            # ============================= TextBiLSTM =================================
        # ============================= AudioBiLSTM =============================
        lengths = mask.sum(dim=1).long().cpu()
        x_pack = nn.utils.rnn.pack_padded_sequence(audio_feature, lengths, batch_first=True, enforce_sorted=False)
        x_pack, _ = self.lstm_net_audio(x_pack)
        x_audio, _ = nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        x_sum = x_audio.sum(dim=1)
        x_audio = x_sum / (lengths.to(x_sum.device).float().unsqueeze(1) + 1e-6)
        audio_feature1 = self.fc_audio(x_audio)
        return text_feature1, audio_feature1

    def forward(self, utterance_text, utterance_audio):
        # Shared-private encoders
        self.shared_private(utterance_text, utterance_audio)
        if not self.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)
            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_a = None
        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator(
            (self.utt_shared_t + self.utt_shared_a) / 2.0)
        # For reconstruction
        self.reconstruct()
        # 1-LAYER TRANSFORMER FUSION
        x = torch.stack((utterance_text, utterance_audio,self.utt_private_t, self.utt_private_a, self.utt_shared_t, self.utt_shared_a), dim=1)
        tgt = self.cls_token.expand(x.size(0), -1, -1)
        src = self.pos_encoder(x)  # [B, F, hidden_dim]
        tgt = self.pos_encoder(tgt)  # [B, 1, hidden_dim]
        # output = self.transformer_encoder(x)
        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=None,
            tgt_mask=None,  # [CLS] token无需掩码
            memory_mask=None,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None
        )  #
        o = self.fc(output.squeeze(1))
        #print(o.shape)
        #x = torch.stack((utterance_text, utterance_audio, self.utt_private_t, self.utt_private_a, self.utt_shared_t,self.utt_shared_a), dim=0)
        h = self.transformer_encoder(x)
        h = torch.cat((h[:, 0, :], h[:, 1, :], h[:, 2, :], h[:, 3, :], h[:, 4, :], h[:, 5, :]), dim=1)
        o = self.fusion(h)
        #print(o.shape)
        return o
