import torch
import torch.nn as nn
from model.audio_gru import AudioBiLSTM
from model.text_Bilstm import TextBiLSTM
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import random
from dataset.get_dataset import get_mdea
import matplotlib.pyplot as plt
prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

def plot_training_history(train_accuracies, val_accuracies,train_losses, val_losses, pickle_path,number):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='train_losses')
    plt.plot(epochs, val_losses, label='val_losses')
    plt.title('trainandval_losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='train_accuracies')
    plt.plot(epochs, val_accuracies, label='val_accuracies')
    plt.title('trainandval_accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(pickle_path + '/training_history'+ str(number) +'.png')
    plt.show()

def save(model, filename):
    # 提取文件目录
    directory = os.path.dirname(filename)
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

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
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test.cpu().numpy(), y_test_pred)
    return np.array([[tp, fp], [fn, tn]])

def model_performance(y_test, y_test_pred_proba):
    """
    Evaluation metrics for network performance.
    """

    # Computing confusion matrix for test dataset
    y_test_pred_proba = y_test_pred_proba.data.max(1, keepdim=True)[1]
    conf_matrix = standard_confusion_matrix(y_test, y_test_pred_proba.numpy())
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_test_pred_proba, conf_matrix


def train(model, epoch, train_idxs,audio_features, audio_targets, optimizer, criterion, config,masks):
    global lr, train_acc
    model.train()
    batch_idx = 0
    total_loss = 0
    correct = 0
    pred = np.array([])
    X_train = audio_features[train_idxs]
    Y_train = audio_targets[train_idxs]
    masks = masks[train_idxs]
    for i in range(0, X_train.shape[0], config['batch_size']):
        if i + config['batch_size'] > X_train.shape[0]:
            x, y, mask = X_train[i:], Y_train[i:], masks[i:]
        else:
            x, y, mask = X_train[i:i + config['batch_size']], Y_train[i:i + config['batch_size']], masks[i:i + config['batch_size']]
        if config['cuda']:
            x, y, mask = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda(), Variable(torch.from_numpy(mask).type(torch.FloatTensor)).cuda()
        else:
            x, y, mask = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True), Variable(torch.from_numpy(y)), Variable(torch.from_numpy(mask).type(torch.FloatTensor))
        optimizer.zero_grad()
        output = model(x, mask)
        # probs = torch.sigmoid(output)
        # pred = (probs > 0.5).float()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        loss = criterion(output, y.long())
        #loss = criterion(output, y.long(), model, config)
        # loss = criterion(output, y.view(-1, 1).float())
        loss.backward()
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()
    avg_loss = total_loss / batch_idx
    train_acc = correct
    train_acc1 = correct / len(X_train)
    print(
        'Train Epoch: {:2d}\t Learning rate: {:.4f}\tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)\n '
        .format(epoch, config['learning_rate'], avg_loss, correct, X_train.shape[0], 100. * correct / X_train.shape[0]))
    return  train_acc1, avg_loss


def evaluate(model, test_idxs, train_idxs,number, audio_features, audio_targets,optimizer, criterion, model_path,  config, masks):
    model.eval()
    batch_idx = 1
    total_loss = 0
    global max_f1, max_acc, min_mae, max_prec, max_rec
    correct_indices = []  # 用于存放正确预测的样本索引
    with torch.no_grad():
        if config['cuda']:
            x, y = Variable(torch.from_numpy(audio_features[test_idxs]).type(torch.FloatTensor), requires_grad=True).cuda(),\
                Variable(torch.from_numpy(audio_targets[test_idxs]).type(torch.LongTensor)).cuda()
            masks = Variable(torch.from_numpy(masks[test_idxs]).type(torch.FloatTensor)).cuda()
        else:
            x, y = Variable(torch.from_numpy(audio_features[test_idxs]).type(torch.FloatTensor), requires_grad=True), \
                Variable(torch.from_numpy(audio_targets[test_idxs]).type(torch.LongTensor))
            masks = Variable(torch.from_numpy(masks[test_idxs]).type(torch.FloatTensor))
        optimizer.zero_grad()
        output = model(x, masks)
        # loss = criterion(output, y.view(-1, 1).float())
        loss = criterion(output, y.long())
        #loss = criterion(output, y.long(),model,config)
        total_loss += loss.item()
        # 获取预测结
        _, predicted = torch.max(output, 1)
        # probs = torch.sigmoid(output)
        # predicted = (probs > 0.5).long().view(-1)
        # 找到正确预测的样本索引
        correct_predictions = (predicted.cpu().numpy() == y.cpu().numpy())
        correct_indices = np.array(test_idxs)[correct_predictions]
        # 对正确预测的样本索引进行排序
        sorted_indices = np.sort(correct_indices)
        # y_test_pred, conf_matrix = model_performance(y, predicted.cpu())
        y_test_pred, conf_matrix = model_performance(y, output.cpu())
        accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
        precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
        recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
        f1_score = 2 * (precision * recall) / (precision + recall)
        avg_loss = total_loss / len(test_idxs)
        print(
            f' Loss: {total_loss:.4f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1_score:.2f} | accuracy : {accuracy:.2f}')
        '''
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1-Score: {}\n".format(f1_score))
        '''
        print('=' * 89)

        if max_f1 < f1_score and train_acc >= len(train_idxs) * 0.70 and f1_score >= 0.50 and accuracy >= 0.50:
            max_f1 = f1_score
            max_acc = accuracy
            max_rec = recall
            max_prec = precision
            mode = 'gru'
            save(model, os.path.join(model_path, 'BiLSTM_vlad{:.2f}_{:.2f}_{:.2f}_{:.2f}_{}'.format(
                                                                                                             precision,
                                                                                                             recall,
                                                                                                             f1_score,
                                                                                                             accuracy,
                                                                                                             number)))
            print(f'预测正确的样本索引：{sorted_indices}')
            print('*' * 64)
            print('model saved: recall: {}\tf1: {}\tacc: {}'.format(recall, f1_score, accuracy))
            print('*' * 64)
    return accuracy, total_loss

def get_param_group(model):
    nd_list = []
    param_list = []
    for name, param in model.named_parameters():
        if 'ln' in name:
            nd_list.append(param)
        else:
            param_list.append(param)
    return [{'params': param_list, 'weight_decay': 1e-4}, {'params': nd_list, 'weight_decay': 0}]

config1 = {
    'num_classes': 2,
    'num_classes1':3,
    'dropout': 0.5,
    'rnn_layers': 2,
    'embedding_size': 88,
    #'batch_size': 2,
    'batch_size': 32,
    'epochs': 60,
    'learning_rate' : 5e-5,
    #'learning_rate': 6e-6,
    'hidden_dims': 256,
    'bidirectional': True,
    'cuda': True,
}
config2 = {
    'num_classes': 2,
    'num_classes1':3,
    'dropout': 0.5,
    'rnn_layers': 2,
    'embedding_size': 1024,
    'batch_size': 32,
    'epochs': 170,
    'learning_rate': 5e-5,
    'hidden_dims': 256,
    'bidirectional': True,
    'cuda': True,
}


def main_audio(number,languages,random_i):
    config = config1
    path_model = os.path.join(prefix, 'efficiency/{}/audio'.format(languages))
    audio_features, text_features, fuse_targets, languages_targets, features, train_idxs, test_idxs, mask = get_mdea(number, languages)
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    pickle_file = os.path.join(prefix, 'premodel/audio/lld/history')
    if not os.path.exists(pickle_file):
        os.makedirs(pickle_file)
    random.shuffle(train_idxs)
    random.shuffle(test_idxs)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    model = AudioBiLSTM(config)
    if config['cuda']:
        model = model.cuda()
    param_group = get_param_group(model)
    optimizer = optim.AdamW(param_group, lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    # 训练模型
    for epoch in range(1, config['epochs'] + 1):
        print(f"random:{random_i}")
        print(f"number: {number}")
        print(f"Epoch {epoch}/{config['epochs']}")
        train_acc, train_loss = train(model, epoch, train_idxs,audio_features, fuse_targets, optimizer, criterion, config,mask)
        test_acc, test_loss = evaluate(model, test_idxs, train_idxs,number,audio_features, fuse_targets, optimizer,
                                                 criterion, path_model,config,mask)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    plot_training_history(train_accuracies, test_accuracies, train_losses, test_losses, pickle_file, number)
def main_text(number,languages,random_i):
    config = config2
    path_model = os.path.join(prefix, 'efficiency/{}/text'.format(languages))
    audio_features, text_features, fuse_targets, languages_targets,fuse_features, train_idxs, test_idxs, mask= get_mdea(number,languages)
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    pickle_file = os.path.join(prefix, 'premodel/text/lld/history')
    if not os.path.exists(pickle_file):
        os.makedirs(pickle_file)
    random.shuffle(train_idxs)
    random.shuffle(test_idxs)
    model = TextBiLSTM(config)
    if config['cuda']:
        model = model.cuda()
    param_group = get_param_group(model)
    optimizer = optim.Adam(param_group, lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()
    # 训练模型
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(1, config['epochs'] + 1):
        print(f"random:{random_i}")
        print(f"number: {number}")
        print(f"Epoch {epoch}/{config['epochs']}")
        train_acc, train_loss = train(model, epoch, train_idxs,text_features,fuse_targets, optimizer, criterion, config,mask)
        test_acc, test_loss = evaluate(model, test_idxs, train_idxs,number, text_features, fuse_targets, optimizer,
                                                 criterion, path_model,config,mask)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    plot_training_history(train_accuracies, test_accuracies, train_losses, test_losses, pickle_file, number)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    i= 44
    max_f1 = -1
    max_acc = -1
    max_rec = -1
    max_prec = -1
    train_acc = -1
    best_loss = -1
    set_seed(i)
    languages = 'zh'
    number = 1
    if number == 1:
        #audio
        main_audio(number,languages,i)
    else:
        #text
        main_text(number,languages,i)