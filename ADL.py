"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn
import numpy as np
#import transforms
import os
import math
import argparse
import torch.utils.data as Data
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
#import pytorch_lightning as pl
from torchvision import transforms
import net_import
import matplotlib.pyplot as plt
import torch.nn.functional as F
import xlrd
import xlutils  
import xlwt
#import vit_fusion
from torchsummary import summary
import net_import
import time
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import mean_squared_error as compare_mse
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def image_show(img,lab):
    plt.figure("Image")
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("real")
    plt.subplot(1, 2, 2)
    plt.imshow(lab)
    plt.title("label")
    plt.show()


'''-------------TEST_PART-----------------------------'''

def c_evaluate(y_actual, y_predicted):
    # print(y_actual.shape,"11111")
    y_actual=y_actual
    y_predicted=y_predicted
   # y_actual=np.array(7*y_actual)
   # y_predicted=np.array(7*y_predicted)
    acc_mse = []
    acc = []
    acc_ssim = []
    f1score = []
    for i in range(0, y_actual.shape[0]):


       # a = mean_squared_error(y_actual[i], y_predicted[i])
        #print(y_actual.shape)
        y_actual[y_actual >= 0.5] = 1
        y_actual[y_actual < 0.5] = 0
        c = compare_ssim(y_actual[i,], y_predicted[i,],data_range=y_actual[i,].max() - y_actual[i,].min())
      #  a = 1 - mean_squared_error(y_actual[i], y_predicted[i])
        a = 1 - compare_mse(y_actual[i], y_predicted[i])

        b =1-np.mean(abs(y_actual[i]- y_predicted[i]))
        acc_mse.append(a)
        acc.append(b)
        acc_ssim.append(c)
    # print(y_actual.shape)

    return [np.mean(acc_mse),np.mean(acc), np.mean(acc_ssim)]


def acc_test_forward(model,test_dataset,y_true):
    y_true=y_true.reshape(y_true.shape[0],1024)
    test_dataset=test_dataset.reshape(test_dataset.shape[0], 32, 32)
    test_dataset = np.expand_dims(test_dataset, axis=1)
    torch_acc_dataset = Data.TensorDataset(torch.Tensor(test_dataset),
                                                    torch.Tensor(test_dataset.reshape(test_dataset.shape[0], 1024)))
    acc_dataloader = Data.DataLoader(
        dataset=torch_acc_dataset,  # torch TensorDataset format
        batch_size=100,  # mini batch size
        shuffle=0, 
        num_workers=0, 
    )
    acc = []
    accmap = []
    predict=[]
    test_num=0
    for (images, labels) in acc_dataloader:
        images=images.to(device)
        pre=model(images).cpu().detach().numpy()
        accmap.append(c_evaluate(pre,y_true[test_num:test_num+100]))
        test_num=test_num+1
        predict.extend(pre)
    print("test acc is {}".format(np.mean(acc)))
    return  accmap,predict

def acc_test_rev(model,test_dataset,y_true):
    y_true=y_true.reshape(y_true.shape[0],1024)
    test_dataset=test_dataset.reshape(test_dataset.shape[0], 150, 150)
    test_dataset = np.expand_dims(test_dataset, axis=1)
    torch_acc_dataset = Data.TensorDataset(torch.Tensor(test_dataset),
                                                    torch.Tensor(y_true.reshape(y_true.shape[0], 1024)))
    acc_dataloader = Data.DataLoader(
        dataset=torch_acc_dataset,  # torch TensorDataset format
        batch_size=500,  # mini batch size
        shuffle=0,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    acc = []
    accmap = []
    predict=[]
    test_num=0
    for (images, labels) in acc_dataloader:
        images=images.to(device)
        pre=model(images).cpu().detach().numpy()
        accmap.append(c_evaluate(pre,y_true[test_num:test_num+500]))
        test_num=test_num+500
        predict.extend(pre)
    print("test acc is {}".format(np.mean(acc)))
    return  accmap,predict


'''-------------TRAIN_PART-----------------------------'''
def train_model(model=0,train_dataloader=0,test_dataloader=0,learning_rate=0,loss=0,work_dir=0,load_path=0,epoch=200):



    load_path = load_path + work_dir+"\\"


    #     load_path = load_path + "self\\"
    if not os.path.exists(load_path):
        os.makedirs(load_path)


    loss_fn =loss# nn.MSELoss()
    # loss_fn = nn.BCEWithLogitsLoss()  #nn.BCELoss()
    early_break = 0
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 添加tensorboard
    writer = SummaryWriter("{}/logs".format(work_dir))
    save_bestloss = 100.0
    last_lost = 100.0
    loss = torch.tensor(0)
    wk = xlwt.Workbook()
    sh = wk.add_sheet("shee1")
    for i in range(epoch):

        early_break = early_break + 1
        print("early_break_num:", early_break)
        if early_break > 10:
            print("now_earlybreak")
            break
        print("-------epoch  {} -------".format(i + 1))
        # 训练步骤
        model.train()
        running_loss = 0
        for step, data in enumerate(train_dataloader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            # 优化器
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar("train_loss", running_loss / len(train_dataloader), i)
        print('train_loss: %.3f  ' %
              (running_loss / (len(train_dataloader))))
        # 测试步骤
        model.eval()
        total_test_loss = 0.0
        total_accuracy = 0.0
        pre = []
        pre_save = 0
        with torch.no_grad():
            for imgs, targets in test_dataloader:
                if torch.cuda.is_available():
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                accuracy =torch.mean(torch.abs(outputs - targets))
                total_accuracy += accuracy
        ave_test_loss = (total_test_loss / len(test_dataloader))
        ave_accuracy = (total_accuracy / (len(test_dataloader)))
        if ave_test_loss - last_lost < 0:
            early_break = 0
        last_lost = ave_test_loss

        if save_bestloss > ave_test_loss:
            with torch.no_grad():
                total_van_accuracy = 0.0
                for imgs, targets in test_dataloader:
                    if torch.cuda.is_available():
                        imgs = imgs.to(device)
                        targets = targets.to(device)
                    outputs = model(imgs)
                    if pre_save == 0:
                        pre = outputs
                        pre_save = 1
                    else:
                        pre = torch.cat((pre, outputs), dim=0)
                    van_accuracy = torch.mean(torch.abs(outputs - targets))
                    total_van_accuracy += van_accuracy
            # np.save(load_path + "predict.npy", pre.to("cpu").numpy())
            sh.write(i + 1, 4, (total_van_accuracy / (len(test_dataloader))).item())
            save_bestloss = ave_test_loss
            torch.save(model, load_path+"/module_best.pth")

            # np.save(load_path + "predict.npy", pre.to("cpu").numpy())
            print("saved epoch {}".format(i + 1))
            print("van_accuracy: {}".format((total_van_accuracy / (len(test_dataloader)))))
            early_break = 0

        sh.write(i + 1, 0, running_loss / len(train_dataloader))
        sh.write(i + 1, 1, ave_test_loss)
        sh.write(i + 1, 2, ave_accuracy.item())

        print("test set Loss: {}".format(ave_test_loss))
        print("test set accuracy: {}".format(ave_accuracy))
        writer.add_scalar("test_loss", ave_test_loss, i)
        writer.add_scalar("test_accuracy", ave_accuracy, i)

 

    wk.save(load_path+ "self.xls")

    writer.close()



def finetune_model(reverse_model=0,train_x=0, train_y=0,val_x=0,val_y=0,learning_rate=0,loss_f=0,load_path=0,epoch=20,num=10000):

    loss_fn = loss_f
    early_break = 0

    opt = torch.optim.AdamW(reverse_model.parameters(), lr=learning_rate)

    save_bestloss = 100.0
    last_lost = 100.0
    loss = torch.tensor(0)
    val_x=torch.tensor(val_x.astype(np.float32))
    val_y=val_y.reshape(val_y.shape[0], 1024)
    train_x=torch.tensor(train_x[-num:].astype(np.float32))
    torch_train_dataset = Data.TensorDataset(train_x,
                                          train_y[-num:].reshape(num,1024))  # TORCH.Tensor and tensor not the same
    train_dataloader = Data.DataLoader(
        dataset=torch_train_dataset,  # torch TensorDataset format
        batch_size=100,  # mini batch size
        shuffle=0,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    if torch.cuda.is_available():
        imgs = val_x.to(device)
        targets = val_y.to(device)
    opt.zero_grad()
    outputs = reverse_model(imgs)
    loss = loss_fn(outputs, targets)
    file_path = load_path+"befor_finetune.txt"
    if os.path.exists(file_path):
      
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(str(loss.item())+ "\n")
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(loss.item()) + "\n")



    for i in range(epoch):

        print("-------epoch  {} -------".format(i + 1))

        running_loss = 0
        reverse_model.train()
        for step, data in enumerate(train_dataloader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                targets = targets.to(device)
            opt.zero_grad()
            outputs = reverse_model(imgs)
            loss = loss_fn(outputs, targets)

            loss.backward()
            opt.step()
            running_loss += loss.item()
        print('finetune_train_loss: %.3f  ' %
              (running_loss / (len(train_dataloader))))
    reverse_model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            imgs = val_x.to(device)
            targets = val_y.to(device)
        opt.zero_grad()
        outputs = reverse_model(imgs)
        loss = loss_fn(outputs, targets)
        file_path = load_path+"after_finetune.txt"
        if os.path.exists(file_path):

            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(str(loss.item())+ "\n")

        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(loss.item()) + "\n")





    return reverse_model


def finetune_forward_model(forward_model,train_x=0, train_y=0,learning_rate=0,loss_f=0,epoch=20,num=10000):
    loss_fn = loss_f
    early_break = 0

    opt = torch.optim.AdamW(forward_model.parameters(), lr=learning_rate)

    save_bestloss = 100.0
    last_lost = 100.0
    loss = torch.tensor(0)
    train_x=train_x[-num:]
    train_x= train_x.view(num, 1, 32, 32)
    torch_train_dataset = Data.TensorDataset(train_x,
                                          train_y[-num:].reshape(num,1024))  # TORCH.Tensor and tensor not the same
    train_dataloader = Data.DataLoader(
        dataset=torch_train_dataset,  # torch TensorDataset format
        batch_size=100,  # mini batch size
        shuffle=0, 
        num_workers=0, 
    )

    for i in range(epoch):

        print("-------epoch  {} -------".format(i + 1))
   
        running_loss = 0
        forward_model.train()
        for step, data in enumerate(train_dataloader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                targets = targets.to(device)
            opt.zero_grad()
            outputs = forward_model(imgs)
            loss = loss_fn(outputs, targets)
            # 优化器

            loss.backward()
            opt.step()
            running_loss += loss.item()
        print('finetune_train_loss: %.3f  ' %
              (running_loss / (len(train_dataloader))))
        # model.eval()

    return forward_model


def train_transmit_model(reverse_model=0,forward_model=0, train_data_in=0,train_data_y=0,train_data_out=0 ,num=0,learning_rate=0, work_dir=0, load_path=0, epoch=20):

    accmap=[]
    load_path=load_path+work_dir
    pre_label =[]
    transmit_step=500
    train_squ_num=int(train_data_in.shape[0]/transmit_step)-42#42  -210
    loss_fn = nn.MSELoss()
    err_ev=0
    err_fix=0

    train_data_out=train_data_out.reshape(train_data_out.shape[0],1024)
    wk = xlwt.Workbook()
    sh = wk.add_sheet("shee1")
    time_begin=time.time()
    for i in range(train_squ_num):
        x_in_now=torch.Tensor(train_data_in[num+i*transmit_step:num+(i+1)*transmit_step])
        x_out_now= torch.Tensor(train_data_out[num+i*transmit_step:num+(i+1)*transmit_step])
        torch_xx_dataset = Data.TensorDataset(x_in_now, x_out_now)  # TORCH.Tensor and tensor not the same
        train_xx_dataloader = Data.DataLoader(
            dataset=torch_xx_dataset,  # torch TensorDataset format
            batch_size=100,  # mini batch size
            shuffle=0, 
            num_workers=0, 
        )
        err_1000 = 0


        reverse_model.eval()
        forward_model.eval()
        for step, data in enumerate(train_xx_dataloader, start=0):
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                targets = targets.to(device)
            with torch.no_grad():
                rev_begin=time.time()
                outputs_pattern = reverse_model(imgs)
                rev_end = time.time()
                outputs_pattern = outputs_pattern.view(outputs_pattern.shape[0], 32, 32).unsqueeze(1)  
                outputs_speckles = forward_model(outputs_pattern)
                rev_end = time.time()


            err_now = loss_fn(outputs_speckles, targets)
            err_1000+=err_now

        err_1000=err_1000

        if i < 5:
            err_ev += err_1000
            if i == 4:
                err_ev = err_ev / 5
                print( err_ev,"now_err_ev")
                sh.write(i + 1, 5, int(err_ev.cpu().detach().numpy()))
                a=err_ev/30
        elif i >= 5:

            print(err_1000,"now_err_1000")
            print(err_ev, "now_err_ev")
            if err_fix==1:
                err_ev = err_ev + 0.8 * (err_1000 - err_ev)
                err_fix=0
            if err_ev>err_1000:
                err_ev = err_ev + 0.5 * (err_1000 - err_ev)
            if err_1000 >err_ev+ a:

                print("now_activate fintune")
                # rev_begin = time.time()
                reverse_model = finetune_model(reverse_model,
                    train_x=train_data_in[:num + i * transmit_step],
                    train_y=torch.tensor(pre_label, dtype=torch.float32),
                    val_x=train_data_in[num + i * transmit_step:num + i * transmit_step+transmit_step],
                    val_y=torch.tensor(train_data_y[num + i * transmit_step:num + i * transmit_step+transmit_step], dtype=torch.float32),
                    learning_rate=learning_rate, loss_f=nn.BCELoss(), load_path=load_path,epoch=10)
                err_fix=1

        reverse_model.eval()
        with torch.no_grad():
            outputs_pattern = reverse_model(x_in_now.to(device)).cpu()
            outputs_pattern[outputs_pattern>0.5]=1
            outputs_pattern[outputs_pattern < 0.5] = 0

        if i == 0:
            pre_label = train_data_y[:num, ].reshape(num,1024)
            tf=torch.Tensor(pre_label)
            pre_label=torch.cat((tf,outputs_pattern),dim=0)
        else:
            pre_label=torch.cat((pre_label,outputs_pattern),dim=0)

    #########################
        outputs_pattern = outputs_pattern.detach().numpy()
        acc_now = c_evaluate(
            train_data_y[num + i * transmit_step:num + (i + 1) * transmit_step].reshape(transmit_step, 1024),
            outputs_pattern)  # 这块有bug
        accmap.append(acc_now)  # mse acc ssim
        # np.append(pre_label,outputs_pattern)

        print("saved epoch {}".format(i + 1))
        print("van_accuracy: {}".format(acc_now[0]))
        sh.write(i + 1, 0, err_1000.cpu().detach().item())
        sh.write(i + 1, 1, acc_now[0])
        sh.write(i + 1, 2, acc_now[1])
        sh.write(i + 1, 3, acc_now[2])

    wk.save(load_path + ".xls")

    np.save(load_path + "predict.npy", pre_label.to("cpu").numpy())




parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default="")        #your data_path
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--work_dir', default='./CNN', help='create model name')

parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
parser.add_argument('--freeze-layers', type=bool, default=True)
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
print(torch.cuda.is_available(), "gpu")
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
def train(work_dir=False,load_path=False,pre_train=0):
    # global reverse_model
    args = parser.parse_args()
    batch_size = args.batch_size
    epoch = args.epoch
    learning_rate = args.lr

    if load_path == False:
        path = args.path
    else:
        savepath=load_path
    num = 10000
    y_train = np.load(load_path + 'pattern.npy')

    x_train = np.load(load_path + "x_32.npy")/ 255
    x_train=x_train.reshape(x_train.shape[0],32,32)
    y_train = y_train.reshape(y_train.shape[0], 32, 32)
    x_train = np.expand_dims(x_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    y_test = y_train[num:num+200, ]
    x_test = x_train[num:num+200, ]
    x_train_big = np.load(load_path + "speckles.npy") / 255
    x_train_big = np.expand_dims(x_train_big, axis=1)
    x_test_big = x_train_big[num:num+200, ]


    if "CNN0" in work_dir:
        

        load_path = load_path + "CNN0\\"

    elif "MLP" in work_dir:

        load_path = load_path + "MLP\\"
        print("now work in MLP")
    elif "U_NET" in work_dir:

        print("now work in U_NET")
        load_path = load_path + "U_NET\\"


    if not os.path.exists(load_path):
        os.makedirs(load_path)

    # summary(model, (1, 32, 32))
    print("x.shape{}".format(y_train.shape))

    torch_dataset = Data.TensorDataset(torch.Tensor(x_train_big[:num]),torch.Tensor(y_train[:num].reshape(num,1024)))  # TORCH.Tensor and tensor not the same
    train_dataloader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=0,
        num_workers=0,
    )

    torch_test_dataset = Data.TensorDataset(torch.Tensor(x_test_big), torch.Tensor(y_test.reshape(y_test.shape[0],1024)))
    test_dataloader = Data.DataLoader(
        dataset=torch_test_dataset,
        batch_size=100,
        shuffle=0,
        num_workers=0,
    )

    torch_dataset = Data.TensorDataset(torch.Tensor(y_train[:num]),
                                       torch.Tensor(x_train[:num].reshape(num,1024)))  # TORCH.Tensor and tensor not the same
    Forward_train_dataloader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=0,
        num_workers=0,
    )

    Forward_torch_test_dataset = Data.TensorDataset(torch.Tensor(y_test), torch.Tensor(x_test.reshape(y_test.shape[0],1024)))
    Forward_test_dataloader = Data.DataLoader(
        dataset=Forward_torch_test_dataset,  # torch TensorDataset format
        batch_size=100,
        shuffle=0,  #
        num_workers=0,  #
    )

    if pre_train==1:

        reverse_model = torch.load(load_path +
                                   "Reverse_model\\" + 'module_best.pth').to(device)
    else:
        reverse_model = net_import.cnn_big(1, 32, ).to(device)

        train_model(model=reverse_model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                learning_rate=learning_rate, loss=nn.BCELoss(), work_dir="Reverse_model", load_path=load_path,
                epoch=200)
        Forward_model = net_import.cnn(1, 32, ).to(device)
        train_model(model=Forward_model, train_dataloader=Forward_train_dataloader,
                    test_dataloader=Forward_test_dataloader,
                    learning_rate=learning_rate, loss=nn.MSELoss(), work_dir="Forward_model", load_path=load_path,
                    epoch=200)
    accmap, predict = acc_test_rev(reverse_model, x_train_big[num:, ], y_train[num:, ])
    accmap = np.array(accmap)
    np.save(load_path + "pre_train_forward_predict.npy", np.array(predict))

    print(np.array(predict).shape)
    wk = xlwt.Workbook()
    sh = wk.add_sheet("shee1")

    for i in range(len(accmap)):
        sh.write(i + 1, 0, 1 * accmap[int(i), 0])
        sh.write(i + 1, 1, 1 * accmap[int(i), 1])

    wk.save(load_path + "pre_train_result.xls")


    #
    reverse_model = torch.load( load_path+"Reverse_model\\" + 'module_best.pth').to(device)         #train reverse_model
    Forward_model = torch.load(load_path+"Forward_model\\" + 'module_best.pth').to(device)          #train Forward_model
    train_transmit_model(reverse_model=reverse_model,forward_model=Forward_model, train_data_in=x_train_big, train_data_y=y_train, train_data_out=x_train,num=num,
                         learning_rate=learning_rate, work_dir="transmit", load_path=load_path, epoch=10)                                                                   #ADL transmit






if __name__ == '__main__':

    train(work_dir="CNN0", load_path=r"your data load_path", pre_train=0)
