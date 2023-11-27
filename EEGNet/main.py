import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from torch.utils.data import DataLoader
from EEGNet import EEGNet
from SelfNet import SelfNet
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from GetData import GetLoader
from scipy.io import savemat
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score, accuracy_score
import seaborn as sns  # 导入seaborn绘图库
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='the classification of valence in DEAP dataset')
parser.add_argument('--nUser', default=15, type=int, help='the number of users')  # DEAP 32  DREAMER 23  SEED 15
parser.add_argument('--nSession', default=3, type=int, help='the number of users')
parser.add_argument('--lr', default=0.0001, type=float, help='the initial learning rate')
parser.add_argument('--epochs', default=300, type=int, help='the epochs of training process')
parser.add_argument('--optimizer', default='SGD', type=str, help='the optimizer: SGD, Adam')
parser.add_argument('--momentum', default=0.9, type=float, help='the momentum of SGD optimizer')
parser.add_argument('--num_classes', default=3, type=int, help='the number of classification categories')
parser.add_argument('--batch_size', default=16, type=int, help='the batch size of DataLoader')
parser.add_argument('--device', default='cuda:0', type=str, help='the number of gpu device')
# parser.add_argument('--data_path', default='../data/EEG/DEAP/eachSub/data/', type=str, help='the path of data')
# parser.add_argument('--label_path', default='../data/EEG/DEAP/eachSub/label/', type=str, help='the path of label')
# parser.add_argument('--data_path', default='../data/EEG/DREAMER/eachSub/data/', type=str, help='the path of data')
# parser.add_argument('--label_path', default='../data/EEG/DREAMER/eachSub/label/', type=str, help='the path of label')
parser.add_argument('--data_path', default='../data/EEG/SEED/data/', type=str, help='the path of data')
parser.add_argument('--label_path', default='../data/EEG/SEED/label/', type=str, help='the path of label')
parser.add_argument('--save_model_path', default='./result/model/', type=str, help='the path of save model')
parser.add_argument('--save_record_path', default='./result/record/', type=str, help='the path of save train val test')
parser.add_argument('--train_pattern', default='train', type=str, help='train pattern or val pattern or test pattern')
parser.add_argument('--CNN_features_path', default='./result/cnn_features/', type=str,
                    help='the path of save cnn cnn_features')
parser.add_argument('--routing_iterations', type=int, default=3)


# 训练和验证
def train():
    # 训练
    net.train()
    all_loss, correct, total, best_acc = 0, 0, 0, 0.0
    with tqdm(trainLoader, ncols=None, file=sys.stdout) as t:
        for batch_idx, (inputs, targets) in enumerate(t):
            t.set_description('train epoch[{}/{}]'.format(epoch + 1, args.epochs))  # 给进度条添加描述
            inputs = inputs.to(device)
            inputs = inputs.to(torch.float32)  # 需要进行数据类型的转换
            targets = targets.to(device)
            targets = targets.long()  # 需要进行数据类型的转换
            optimizer.zero_grad()
            outputs = net(inputs)

            # 计算loss以及优化器优化模型
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 计算loss和以及总样本和预测样本数，方便实时计算精度
            all_loss += loss.item()
            predict_y = torch.max(outputs, dim=1)[1]
            total += targets.size(0)
            correct += predict_y.eq(targets).sum().item()
            train_loss = all_loss / (batch_idx + 1)
            train_acc = correct / total

            # 打印训练相关数值
            t.set_postfix_str(
                'train_loss:{:.3f}, train_acc:{:.3f}'.format(train_loss, train_acc))

    # 验证
    net.eval()
    all_loss, correct, total = 0, 0, 0
    # 不需要进行梯度下降求导
    with torch.no_grad():
        with tqdm(valLoader, ncols=None, file=sys.stdout) as t:
            for batch_idx, (inputs, targets) in enumerate(t):
                inputs = inputs.to(device)
                inputs = inputs.to(torch.float32)  # 需要进行数据类型的转换
                targets = targets.to(device)
                targets = targets.long()  # 需要进行数据类型的转换
                outputs = net(inputs)

                loss = criterion(outputs, targets)
                all_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                total += targets.size(0)
                correct += predict_y.eq(targets).sum().item()
                val_loss = all_loss / (batch_idx + 1)
                val_acc = correct / total
                t.set_postfix_str(
                    'val_loss:{:.3f}, val_acc:{:.3f}'.format(val_loss, val_acc))
            # 模型保存
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net, save_path.format(i + 1, session + 1))
    train_loss_epoch.append(train_loss)
    train_acc_epoch.append(train_acc)
    val_loss_epoch.append(val_loss)
    val_acc_epoch.append(val_acc)


# 测试
def test(loader):
    print("====================测试到第{}个受试者的第{}次实验====================".format(i + 1, session + 1))
    net = torch.load(save_path.format(i + 1, session + 1))  # 加载精度最高的模型
    net.eval()
    all_loss, correct, total = 0, 0, 0

    y_target = []
    y_predict = []

    with torch.no_grad():
        with tqdm(loader, ncols=None, file=sys.stdout) as t:
            for batch_idx, (inputs, targets) in enumerate(t):
                inputs = inputs.to(device)
                inputs = inputs.to(torch.float32)  # 需要进行数据类型的转换
                targets = targets.to(device)
                targets = targets.long()  # 需要进行数据类型的转换
                outputs = net(inputs)

                # 只有正常的测试集需要计算下面的loss以及精度等等，如果只需要得到全连接层的结果的话，只需要计算到此处即可
                loss = criterion(outputs, targets)
                all_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                total += targets.size(0)
                correct += predict_y.eq(targets).sum().item()
                test_loss = all_loss / (batch_idx + 1)
                test_acc = correct / total
                t.set_postfix_str(
                    'test_loss:{:.3f}, test_acc:{:.3f}'.format(test_loss, test_acc))

                y_target.append(targets.cpu().numpy())
                y_predict.append(predict_y.cpu().numpy())
        test_acc_list.append(test_acc)


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    device = args.device

    # 所有的loss和acc
    train_loss_list = list()
    train_acc_list = list()
    val_loss_list = list()
    val_acc_list = list()
    test_acc_list = list()
    test_confusion = list()
    test_f1 = list()
    test_re = list()
    test_pr = list()

    # 记录训练过程中train_loss,acc以及val_loss,acc以及test_acc
    trainValRecord = np.zeros((4, args.nSession * args.nUser, args.epochs))
    index = 0    # 保存数据的下标，从0到45

    for i in range(args.nUser):
        for session in range(args.nSession):  # 实验的次数
            print("训练到第{}个受试者的第{}次实验".format(i + 1, session + 1))

            # 存储每个epoch的loss和acc
            train_loss_epoch = list()
            train_acc_epoch = list()
            val_loss_epoch = list()
            val_acc_epoch = list()

            # EEG信号
            data = np.load(args.data_path + 's{}-{}.npy'.format(i + 1, session + 1))

            label = np.load(args.label_path + 's{}-{}.npy'.format(i + 1, session + 1))

            # label处理，交叉熵只支持0,1,2 这样的
            for idx in range(len(label)):
                label[idx] = label[idx] + 1

            # 训练集和验证集以及测试集的划分
            # x_train, x_test, y_train, y_test = train_test_split(norm_data, label, test_size=0.1, random_state=None)
            x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=None)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=None)

            # 使用重写的DataSet类将数据和标签进行整合
            train_dataset = GetLoader(x_train, y_train)
            val_dataset = GetLoader(x_val, y_val)
            test_dataset = GetLoader(x_test, y_test)

            # 使用DataLoader加载DataSet类整合之后的数据
            trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
            testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

            # 网络实例化以及损失和优化器定义
            net = SelfNet(args.num_classes)  # 三分类
            net = net.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

            # 参数定义
            save_path = args.save_model_path + 's{}-{}-bestModel.pth'
            for epoch in range(args.epochs):
                train()  # 训练
            print("====================Training Finished====================")
            test(testLoader)  # 测试

            trainValRecord[0][index] = np.array(train_loss_epoch)
            trainValRecord[1][index] = np.array(val_loss_epoch)
            trainValRecord[2][index] = np.array(train_acc_epoch)
            trainValRecord[3][index] = np.array(val_acc_epoch)
            testRecord = np.array(test_acc_list)
            np.save(args.save_record_path + 'trainValRecord.npy', trainValRecord)
            np.save(args.save_record_path + 'testRecord.npy', testRecord)
            index = index + 1
