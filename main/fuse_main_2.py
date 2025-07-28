import torch
import torch.nn as nn
from model.audio_gru import AudioBiLSTM
from model.text_Bilstm import TextBiLSTM
from model.fuse_net import Fusion_MMS
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pickle
import random
import itertools
from dataset.get_dataset import get_mdea
import matplotlib.pyplot as plt
from utils import to_gpu, CMD, MSE, DiffLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

def plot_training_history(train_accuracies, val_accuracies,train_losses, val_losses, pickle_path,number):
    epochs = range(1, len(train_losses) + 1)
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='train_losses')
    plt.plot(epochs, val_losses, label='val_losses')
    plt.title('trainandval_losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='train_accuracies')
    plt.plot(epochs, val_accuracies, label='val_accuracies')
    plt.title('trainandval_accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #保存图像
    plt.savefig(pickle_path + '/training_history'+ str(number) +'.png')
    plt.show()

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

device = torch.device("cpu")
def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])


def model_performance(y_test, y_test_pred_proba):
    """
    Evaluation metrics for network performance.
    """
    # y_test_pred = y_test_pred_proba.data.max(1, keepdim=True)[1]
    y_test_pred = y_test_pred_proba

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_test_pred, conf_matrix

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.diff_weight = 0.3
        self.sim_weight = 0.5
        self.recon_weight = 1.0
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        #self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        self.use_cmd_sim = True

    def get_domain_loss(self):
        if self.use_cmd_sim:
            return 0.0
        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_a = self.model.domain_label_a
        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0] * domain_pred_t.size(0)))
        domain_true_a = to_gpu(torch.LongTensor([1] * domain_pred_a.size(0)))
        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t,domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t,domain_true_a), dim=0)
        return self.domain_loss_criterion(domain_pred, domain_true)
    def get_cmd_loss(self, ):

        if not self.use_cmd_sim:
            return 0.0
        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
        loss = loss
        return loss

    def get_diff_loss(self):
        shared_t = self.model.utt_shared_t
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_a = self.model.utt_private_a
        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_a, shared_a)
        # Across privates
        loss += self.loss_diff(private_a, private_t)
        return loss
    def get_recon_loss(self):
        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss / 2.0
        return loss
    def forward(self,pred, target, model):
        #target = target.view(-1, 1).float()
        self.model = model
        cls_loss = self.criterion(pred, target)
        diff_loss = self.get_diff_loss()
        domain_loss = self.get_domain_loss()
        recon_loss = self.get_recon_loss()
        cmd_loss = self.get_cmd_loss()
        if self.use_cmd_sim:
            similarity_loss = cmd_loss
        else:
            similarity_loss = domain_loss

        loss = cls_loss + \
               self.diff_weight * diff_loss + \
               self.sim_weight * similarity_loss + \
               self.recon_weight * recon_loss
        return loss


def train(model,epoch, train_idxs,fuse_features, fuse_targets,masks, optimizer, criterion, config):
    global max_train_acc, train_acc
    model = model.to(device)
    model.train()
    batch_idx = 0
    total_loss = 0
    correct = 0
    X_train = []
    Y_train = []
    Mask_train = []
    for idx in train_idxs:
        X_train.append(fuse_features[idx])
        Y_train.append(fuse_targets[idx])
        Mask_train.append(masks[idx])
    for i in range(0, len(X_train), config['batch_size']):
        if i + config['batch_size'] > len(X_train):
            x, y = X_train[i:], Y_train[i:]
            mask = Mask_train[i:]
        else:
            x, y = X_train[i:(i + config['batch_size'])], Y_train[i:(i + config['batch_size'])]
            mask = Mask_train[i:(i + config['batch_size'])]
        if config['cuda']:
            x_text = []
            x_audio = []
            for ele in x:
                x_text.append(ele[1])
                x_audio.append(ele[0])
            mask, x_text, x_audio = np.array(mask), np.array(x_text), np.array(x_audio)
            x_text, x_audio = Variable(torch.tensor(x_text).type(torch.FloatTensor),requires_grad=False).to(device), Variable(
                torch.tensor(x_audio).type(torch.FloatTensor), requires_grad=False).to(device)
            y = Variable(torch.tensor(y).type(torch.LongTensor), requires_grad=False).to(device)
            mask = Variable(torch.tensor(mask).type(torch.FloatTensor), requires_grad=False).to(device)
        else:
            x_text = []
            x_audio = []
            for ele in x:
                x_text.append(ele[1])
                x_audio.append(ele[0])
                mask = mask
                x_text, x_audio = Variable(torch.tensor(x_text).type(torch.FloatTensor),
                                       requires_grad=False), Variable(
                torch.tensor(x_audio).type(torch.FloatTensor), requires_grad=False)
                y = Variable(torch.tensor(y).type(torch.LongTensor), requires_grad=False)
                mask = Variable(torch.tensor(mask).type(torch.FloatTensor), requires_grad=False)
        # 将模型的参数梯度设置为0
        optimizer.zero_grad()
        text_feature, audio_feature = model.pretrained_feature(x_text,x_audio,mask)
        #text_feature = torch.from_numpy(ss.fit_transform(text_feature.numpy()))
        #audio_feature = torch.from_numpy(ss.fit_transform(audio_feature.numpy()))
        #at_feature = torch.from_numpy(ss.fit_transform(at_feature.numpy()))
        concat_x = torch.cat((text_feature, audio_feature), dim=1).to(device)
        # dot_x = text_feature.mul(audio_feature)
        # add_x = text_feature.add(audio_feature)
        output = model(text_feature, audio_feature).to(device)
        pred = output.data.max(1, keepdim=True)[1]
        #probs = torch.sigmoid(output)
        #pred = (probs > 0.5).float()
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        #loss = criterion(output, y.view(-1, 1).float())
        #loss = criterion(text_feature, audio_feature,output, y,model,config)
        loss = criterion(output, y, model)
        # 后向传播调整参数
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        batch_idx += 1
        # loss.item()能够得到张量中的元素值
        total_loss += loss.item()
    avg_loss = total_loss / batch_idx
    cur_loss = total_loss
    max_train_acc = correct
    train_acc = correct
    train_acc1 = correct / len(train_idxs)
    print('Train Epoch: {:2d}\t Learning rate: {:.4f}\tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)\n '.format(
        epoch, config['learning_rate'], cur_loss / batch_idx, correct, len(X_train),
                                        100. * correct / len(X_train)))
    return train_acc1,avg_loss


def evaluate(model, test_idxs, fold, train_idxs, fuse_features, fuse_targets,masks,optimizer, criterion,path_model,config):
    model.eval()
    batch_idx = 0
    total_loss = 0
    X_test = []
    Y_test = []
    all_preds = []
    all_targets = []
    Mask_test = []
    for idx in test_idxs:
        X_test.append(fuse_features[idx])
        Y_test.append(fuse_targets[idx])
        Mask_test.append(masks[idx])
    global max_train_acc, max_acc, max_f1
    for i in range(0, len(X_test), config['batch_size']):
        if i + config['batch_size'] > len(X_test):
            x, y = X_test[i:], Y_test[i:]
            mask = Mask_test[i:]
        else:
            x, y = X_test[i:(i + config['batch_size'])], Y_test[i:(i + config['batch_size'])]
            mask = Mask_test[i:(i + config['batch_size'])]
        if config['cuda']:
            x_text = []
            x_audio = []
            for ele in x:
                x_text.append(ele[1])
                x_audio.append(ele[0])
            mask, x_text, x_audio = np.array(mask), np.array(x_text), np.array(x_audio)
            x_text, x_audio = Variable(torch.tensor(x_text).type(torch.FloatTensor),
                                       requires_grad=False).to(device), Variable(
                torch.tensor(x_audio).type(torch.FloatTensor), requires_grad=False).to(device)
            y = Variable(torch.tensor(y).type(torch.LongTensor), requires_grad=False).to(device)
            mask = Variable(torch.tensor(mask).type(torch.FloatTensor), requires_grad=False).to(device)
        else:
            x_text = []
            x_audio = []
            for ele in x:
                x_text.append(ele[1])
                x_audio.append(ele[0])
                mask = mask
                x_text, x_audio = Variable(torch.tensor(x_text).type(torch.FloatTensor),
                                           requires_grad=False), Variable(
                    torch.tensor(x_audio).type(torch.FloatTensor), requires_grad=False)
                y = Variable(torch.tensor(y).type(torch.LongTensor), requires_grad=False)
                mask = Variable(torch.tensor(mask).type(torch.FloatTensor), requires_grad=False)
        with torch.no_grad():
            text_feature, audio_feature = model.pretrained_feature(x_text, x_audio, mask)
            concat_x = torch.cat((text_feature, audio_feature), dim=1)
            output = model(text_feature, audio_feature).to(device)
        #loss = criterion(output,y.view(-1, 1).float())
        #loss = criterion(text_feature, audio_feature,output, y, model,config)
        loss = criterion(output,y, model)
        pred = output.data.max(1, keepdim=True)[1]
        #probs = torch.sigmoid(output)
        #pred = (probs > 0.5).float()  # 二分类预测
        total_loss += loss.item()
        batch_idx += 1
        all_preds.append(pred.cpu())
        all_targets.append(y.view(-1, 1).float().cpu())
    y_test_pred = torch.cat(all_preds).view(-1)
    y_test = torch.cat(all_targets).view(-1)
    y_test_pred, conf_matrix = model_performance(y_test, y_test_pred)
    # custom evaluation metrics
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    avg_loss = total_loss / batch_idx
    print(
        f' Loss: {avg_loss:.4f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1_score:.2f} | accuracy : {accuracy:.2f}')
    '''
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}\n".format(f1_score))
    '''
    print('=' * 89)

    if max_f1 <= f1_score and max_train_acc >= len(train_idxs) * 0.80 and f1_score >= 0.50 and accuracy >= 0.50:
        max_f1 = f1_score
        max_acc = accuracy
        save(model, os.path.join(path_model, 'fuse_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{}'.format(precision,recall,max_f1,accuracy, fold)))
        print('*' * 64)
        print('model saved: f1: {}\tacc: {}'.format(max_f1, max_acc))
        print('*' * 64)
    return accuracy, avg_loss


config_fuse = {
    'num_classes': 2,
    'text_embed_size': 1024,
    'audio_embed_size': 88,
    'text_hidden_dims': 256,
    'audio_hidden_dims': 256,
    'hidden_mha_dim': 128,
    'hidden_at_dims': 256,
    'number_heads': 2,
    'dropout': 0.3,
    'rnn_layers': 2,
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 1e-4,
    'hidden_dims': 256,
    'bidirectional': True,
    'cuda': True,
    'lambda': 1e-4,
    'activation': 'relu',
    'use_cmd_sim': True,
    'rnncell': 'lstm',
}

def main1(number,languages):
    config = config_fuse
    path_model = os.path.join(prefix, 'premodel/DAIC_1/Audio/lld')
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    pickle_file = os.path.join(prefix, 'premodel/DAIC_1/Audio/lld/history')
    if not os.path.exists(pickle_file):
        os.makedirs(pickle_file)

    # english
    #audio_model_paths = os.path.join(prefix,'efficiency/{}/audio/BiLSTM_vlad_0.62_0.56_0.59_0.60_2.pt'.format(languages))
    #text_model_paths = os.path.join(prefix, 'efficiency/{}/text/BiLSTM_256_0.79_0.77_0.78_0.78_2.pt'.format(languages))
    #path_model = os.path.join(prefix, '{}/Fuse_MMS'.format(languages))
    # chinese
    audio_model_paths = os.path.join(prefix,'efficiency/{}/audio/BiLSTM_vlad_0.62_0.56_0.59_0.60_2.pt'.format(languages))
    text_model_paths = os.path.join(prefix, 'efficiency/{}/text/BiLSTM_vlad_0.61_0.71_0.65_0.62_2.pt'.format(languages))
    path_model = os.path.join(prefix, '{}/Fuse_MMS'.format(languages))
    # german
    #audio_model_paths = os.path.join(prefix,'efficiency/{}/audio/BiLSTM_vlad_0.62_0.56_0.59_0.60_2.pt'.format(languages))
    #text_model_paths = os.path.join(prefix,'efficiency/{}/text/BiLSTM_vlad_0.58_0.72_0.65_0.60_2.pt'.format(languages))
    #path_model = os.path.join(prefix, '{}/Fuse_MMS'.format(languages))
    audio_features, text_features, fuse_targets, languages_targets, fuse_features, train_idxs, test_idxs, mask = get_mdea(number,languages)
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    print(f'train_idxs: {len(train_idxs)}, test_idxs: {len(test_idxs)}')
    random.shuffle(train_idxs)
    random.shuffle(test_idxs)
    model = Fusion_MMS(config)
    print(model)
    if config['cuda']:
        model.to(device)
    text_lstm_model = torch.load(text_model_paths).to(device)
    audio_lstm_model = torch.load(audio_model_paths).to(device)
    print(text_lstm_model)
    print(audio_lstm_model)
    model_state_dict = {}
    model.load_state_dict(audio_lstm_model.state_dict(), strict=False)
    model.load_state_dict(text_lstm_model.state_dict(), strict=False)
    model.load_state_dict(model_state_dict, strict=False)
    # model.load_state_dict(fusion_model.state_dict(), strict=False)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    # optimizer = optim.SG D(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    # optimizer = optim.Adam(model.parameters())
    #criterion = nn.BCEWithLogitsLoss()
    criterion = loss()
    #criterion = MyLoss()
    # 训练模型
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(1, config['epochs'] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")
        #train_acc, train_loss = train(model, epoch, train_idxs, train_features, train_labels, train_mask, optimizer, criterion, config)
        #test_acc, test_loss = evaluate(model, test_idxs, number, train_idxs, test_features, test_labels, test_mask,
        #                               optimizer, criterion, path_model, config)
        train_acc, train_loss = train(model, epoch, train_idxs, fuse_features, fuse_targets, mask, optimizer, criterion, config)
        test_acc, test_loss = evaluate(model, test_idxs, number, train_idxs, fuse_features, fuse_targets, mask,
                                       optimizer, criterion, path_model, config)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    # 调用可视化函数
    plot_training_history(train_accuracies, test_accuracies, train_losses, test_losses, pickle_file, number)
    print("Done!")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == '__main__':
    set_seed(44)
    max_f1 = -1
    max_acc = -1
    max_rec = -1
    max_prec = -1
    train_acc = -1
    languages = 'zh'
    number = 2
    main1(number,languages)