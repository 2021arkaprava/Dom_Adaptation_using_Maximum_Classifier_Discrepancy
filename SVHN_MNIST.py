
import numpy as np
import scipy
import tensorflow_datasets as tfds
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from networks import *
from data_prepare import *
import pandas as pd


def main():

    batch_size = 16


    SVHN_transform = transforms.Compose(
        [transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    MNIST_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
    x_source = torchvision.datasets.SVHN(root='./SVHN', split = 'train',
                                            download=False, transform=SVHN_transform)

    x_source_test = torchvision.datasets.SVHN(root='./SVHN', split = 'test',
                                            download=False, transform=SVHN_transform)


    #test_dataset = torchvision.datasets.SVHN(root='./SVHN', split = 'test',
                                        #download=True, transform=transform)

    #x_source_loader = torch.utils.data.DataLoader(x_source, batch_size=batch_size,
                                           # shuffle=True)

    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            # shuffle=False)

    x_target = torchvision.datasets.MNIST(root=' ', train = True,
                                            download=False, transform=MNIST_transform)

    x_target_test = torchvision.datasets.MNIST(root=' ', train = False,
                                            download=False, transform=MNIST_transform)


    #print(x_target[0][0].shape)
    #x_target = torch.cat((x_target, x_target, x_target), -1)

    #x_target_loader = torch.utils.data.DataLoader(x_target, batch_size=batch_size,
                                            #shuffle=True)


    #svhn_train_im, label = train_dataset[0]
    #print(train_dataset.shape)


    




    G = Feature()
    C1 = Predictor()
    C2 = Predictor()
    


    criterion = nn.CrossEntropyLoss().cuda()
   

    opt_g = optim.Adam(G.parameters(), lr = 2e-4, weight_decay=0.0005)
    opt_c1 = optim.Adam(C1.parameters(), lr = 2e-4, weight_decay=0.0005)
    opt_c2 = optim.Adam(C2.parameters(), lr = 2e-4, weight_decay=0.0005)



    train_loader = UnalignedDataLoader()
    train_loader.initialize(x_source, x_target, batch_size, batch_size)
    dataset = train_loader.load_data()
    test_loader = UnalignedDataLoader()
    test_loader.initialize(x_source_test, x_target_test, batch_size, batch_size)
    dataset_test = test_loader.load_data()

    #print(test_loader.get_len())
    
    epochs = 50
    n = 2

    logger_file_train = 'epochs_%s_n_%s.txt' % (epochs, n)
    logger_file_test = 'epochs_%s_n_%s_test.txt' % (epochs, n)

    val_interval = 150
    logs_interval = 50

    checkpoint_dir = 'Model_Weights'

    G = torch.load('%s/epoch_49_n_2_G.pt' % (checkpoint_dir))
    C1 = torch.load('%s/epoch_49_n_2_C1.pt' % (checkpoint_dir))
    C2 = torch.load('%s/epoch_49_n_2_C2.pt' % (checkpoint_dir))

    G.cuda()
    C1.cuda()
    C2.cuda()
    df = pd.DataFrame({'Epoch' : [],
                        'Batch_id' : [],
                        'Sup loss 1' : [],
                        'Sup loss 2' : [],
                        'Discrepancy loss' : []})

    df_test = pd.DataFrame({'Classifier 1' : [],
                            'Classifier 2' : []})

    
    for epoch in range(epochs):
        
        
        for batch_id, data in enumerate(dataset):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            #print(img_t.shape)
            reset_grad(opt_g, opt_c1, opt_c2)


            label_s = label_s.cuda()
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            emb_s = G(img_s)
            out_s1 = C1(emb_s)
            out_s2 = C2(emb_s)

            loss_s1 = criterion(out_s1, label_s)
            loss_s2 = criterion(out_s2, label_s)

            loss_s = loss_s1 + loss_s2
            
            loss_s.backward()

            opt_g.step()
            opt_c1.step()
            opt_c2.step()

            reset_grad(opt_g, opt_c1, opt_c2)





            emb_s = G(img_s)
            out_s1 = C1(emb_s)
            out_s2 = C2(emb_s)

            loss_s1 = criterion(out_s1, label_s)
            loss_s2 = criterion(out_s2, label_s)

            loss_s = loss_s1 + loss_s2


            emb_t = G(img_t)
            out_t1 = C1(emb_t)
            out_t2 = C2(emb_t)

            loss_dis = discrepancy(out_t1, out_t2)
            #loss_t2 = criterion(out_t2, label_s)

            loss_cls = loss_s - loss_dis

            loss_cls.backward()

            opt_c1.step()
            opt_c2.step()

            reset_grad(opt_g, opt_c1, opt_c2)

            



            for i in range(n):
                emb_t = G(img_t)
                out_t1 = C1(emb_t)
                out_t2 = C2(emb_t)

                loss_dis = discrepancy(out_t1, out_t2)

                loss_dis.backward()

                opt_g.step()

                reset_grad(opt_g, opt_c1, opt_c2)

            #print(loss_s1.detach().cpu().item(), loss_s2.detach().cpu().item(), loss_dis.detach().cpu().item())
            if batch_id % logs_interval == 0:
                print('Epoch : {}, Batch_id : {}, \t sup_loss1 : {}, \t sup_loss2 : {}, disc_loss : {}'.format(epoch, batch_id, loss_s1.detach().cpu().item(), loss_s2.detach().cpu().item(), loss_dis.detach().cpu().item()))
                df.loc[len(df)] = [epoch + 1, 
                                   batch_id,
                                   loss_s1.item(),
                                   loss_s2.item(),
                                   loss_dis.item()]
                df.to_csv('logs_MCD/Training_logs.csv')
                #record = open(logger_file_train, 'a')
                #record.write('Epoch : %s, Batch_id : %s, \t sup_loss1 : %s, \t sup_loss2 : %s, disc_loss : %s \n' % (epoch, batch_id, loss_s1.detach().cpu().item(), loss_s2.detach().cpu().item(), loss_dis.detach().cpu().item()))
                #record.close()
        
        G.eval()
        C1.eval()
        C2.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        size = 0

        for batch_id_test, test_data in enumerate(dataset_test):
            img = test_data['T']
            label = test_data['T_label']

            img, label = img.cuda(), label.cuda()
            emb = G(img)
            out_1 = C1(emb)
            out_2 = C2(emb)

            #test_loss += F.nll_loss(out_1, label).data[0]
            _,pred1 = torch.max(out_1,1)
            _,pred2 = torch.max(out_2,1)

            k = label.data.size()[0]

            correct1 += sum(label==pred1)
            #pred1.eq(label.data).cpu().sum()
            correct2 += sum(label==pred2)

            size += k

        #test_loss /= size

        print(
            '\nTest set: Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%)  \n'.format(
                correct1, size,
                100. * correct1 / size, correct2, size, 100. * correct2 / size))
        
        
        torch.save(G, '%s/epoch_%s_n_%s_G.pt' % (checkpoint_dir, epoch, n))
        torch.save(C1, '%s/epoch_%s_n_%s_C1.pt' % (checkpoint_dir, epoch, n))
        torch.save(C2, '%s/epoch_%s_n_%s_C2.pt' % (checkpoint_dir, epoch, n))

        #record = open(logger_file_test, 'a')
        #record.write('%s %s\n' % (float(correct1) / size, float(correct2) / size))
        #record.close()

        df_test.loc[len(df_test)] = [float(correct1) / size,
                                     float(correct1) / size]
        df_test.to_csv('logs_MCD/Test_logs.csv')





        








if __name__ == '__main__':
    main()
        