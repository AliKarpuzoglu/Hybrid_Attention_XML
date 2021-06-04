
import os
import argparse
import math
import numpy as np
import timeit

import scipy

import data_helpers
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from torchsummary import summary

# In[3]:
from hybrid_xml import Hybrid_XML, calculate_test_val, load_glove_embeddings, use_cuda, things, load_data, \
    sequence_length, vocab_size, batch_size, epoch, calculate_scores, ndcg_k, precision_k, arrange_order, \
    hungarian_loss, binary_array_to_graph
from set_transformer.models import SetTransformer

def calculate_test_val( val_loader,logging=False,mode='validate'):
    pre_acc = 0.0
    predictions = []
    average_precision_arr = []

    test_loss = 0
    test_ap = 0
    test_p1, test_p3, test_p5 = 0, 0, 0
    test_ndcg1, test_ndcg3, test_ndcg5 = 0, 0, 0
    model.eval()
    for i, (data, labels) in enumerate(tqdm(val_loader)):

        data = data.cuda()
        labels = labels.cuda()
        labels =labels.float()
        pred = model(data)
        loss = hungarian_loss(pred, labels)

        # loss=criterion(pred,labels.float())/pred.size(0)

        # 计算metric
        labels_cpu = labels.data.cpu()
        labels_cpu_previous = np.copy(labels_cpu)
        pred_cpu = pred.data.cpu()
        pred_cpu_previous = np.copy(pred_cpu)

        labels_cpu, pred_cpu = arrange_order(np.copy(labels_cpu.numpy()), np.copy(pred_cpu.numpy()))
        # if not np.array_equal(labels_cpu_previous,labels_cpu):
            # print("arranged")
        if not np.array_equal(pred_cpu,pred_cpu_previous):
            print('prediction reordered? etwas cringe brudi')
            exit()
        labels_cpu = torch.from_numpy(labels_cpu)
        pred_cpu = torch.from_numpy(pred_cpu)
        _p1, _p3, _p5 = precision_k(pred_cpu.topk(k=5)[1].numpy(), labels_cpu.numpy(), k=[1, 3, 5])
        test_p1 += _p1
        test_p3 += _p3
        test_p5 += _p5

        _ndcg1, _ndcg3, _ndcg5 = ndcg_k(pred_cpu.topk(k=5)[1].numpy(), labels_cpu.numpy(), k=[1, 3, 5])
        test_ndcg1 += _ndcg1
        test_ndcg3 += _ndcg3
        test_ndcg5 += _ndcg5
        ap,ap_arr,f1_macro, f1_micro= calculate_scores(labels_cpu.numpy(), pred_cpu.numpy())
        average_precision_arr.extend(ap_arr)
        test_ap += float(ap)
        predictions.extend(pred_cpu)

        test_loss += float(loss)
    batch_num = i + 1
    test_loss /= batch_num
    test_ap /=batch_num
    test_p1 /= batch_num
    test_p3 /= batch_num
    test_p5 /= batch_num

    test_ndcg1 /= batch_num
    test_ndcg3 /= batch_num
    test_ndcg5 /= batch_num

    print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (test_p1, test_p3, test_p5))
    print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg1, test_ndcg3, test_ndcg5))
    return predictions,average_precision_arr
valid_dataset = data_helpers.get_valid_df()
noises = [0,2]#,3,10]
lrs = [0.0005]#0.001,0.0005]
pred_ratio_list = []
epoch = 15
for lr in lrs:
    for noise in noises:
        for i in range(5):
            print('-'*50)
            print('Loading data...'); start_time = timeit.default_timer()
            data_path = '../MasterThesis/one_to_4_objects_shuffled_order '+str(noise) +' noise.json'
            train_loader, test_loader, val_loader, vocabulary, X_tst, Y_tst, X_trn,Y_trn, X_val, Y_val= load_data(data_path,sequence_length,vocab_size,batch_size,split=i)
            print('Process time %.3f (secs)\n' % (timeit.default_timer() - start_time))

            # load glove
            pretrain = 'glove'
            embedding_dim = 300
            if pretrain == 'glove':
                # input word2vec file path
                file_path = os.path.join('data', 'glove.6B.%dd.txt' % (embedding_dim))
                embedding_weights = load_glove_embeddings(file_path, vocabulary, embedding_dim)

            runname = "1_4 obj |split " + str(i) + " out of 5| "+str(noise)+" noise |"+ str(epoch) + " epochs| lr: " +str(lr)
            # writer = SummaryWriter(comment=runname)

            model = Hybrid_XML(num_labels=76, vocab_size=30001, embedding_size=300, embedding_weights=embedding_weights,
                               max_seq=500, hidden_size=256, d_a=256)  # ,label_emb=label_emb)
            model.load_state_dict(torch.load("models/"+ runname+' model'))
            model.eval()
            # model.load('./rcv_log/rcv_9.pth')
            if use_cuda:
                model.cuda()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=4e-5)

            # if not os.path.isdir('./rcv_log'):
            #     os.makedirs('./rcv_log')
            # trace_file='./rcv_log/trace_rcv.txt'

            # visualize_x =  X_tst[:5]
            # visualize_y = Y_tst[:5]
            predictions =[]
            print("开始进行validation")
            print("test set")
           # calculate_test_val(test_loader,mode="test",logging=True)
            print("validation set (human)")

            predictions,average_precision_arr = calculate_test_val(val_loader,mode="validation")
            # calculate_test_val(val_loader,logging=True,mode='validate')
            ind_largest = np.argpartition(average_precision_arr, -5)[-5:]
            ind_smallest = np.argpartition(average_precision_arr, 5)[:5]
            print("best 5")
            number_objects_pred = [sum([obj[18].item()>0.3 , obj[37].item()>0.3 , obj[56].item()>0.3 , obj[75].item()>0.3]) for obj in predictions]#[obj[18].item() + obj[37].item() + obj[56].item() + obj[75].item() for obj in predictions]
            number_objects_real = [len(valid_dataset[i]['catgy']) for i in range(len(valid_dataset))]
            print(f"noise{noise}")
            correct_predicted_number = np.sum(np.array(number_objects_pred) == np.array(number_objects_real))
            print(f"we predicted the amount of objects  correctly : {correct_predicted_number} times out of {len(number_objects_real)}")
            pred_ratio_list.append(correct_predicted_number)
            # for i in ind_largest:#range(len(predictions)):
            #     number_objects = int(len(predictions[0]) / 19)
            #     objects =[0]*len(predictions[0])
            #     for l,obj in enumerate(valid_dataset[i]['catgy']):
            #         for index in obj:
            #             objects[l*19+index]=1
            #     objects = np.array(objects).reshape(number_objects, 19)
            #     pred_objects_arr = predictions[i].reshape(number_objects, 19)
            #
            #     epsilon = 0.35
            #     described_objects = [[a for a, b in zip(things, objects[i]) if b] for i in range(number_objects)]
            #     pred_objects = [[a for a, b in zip(things, pred_objects_arr[i]) if b>epsilon] for i in range(number_objects)]
            #     print(f"ap: {average_precision_arr[i]}")
            #     print("--------------")
            #     print(valid_dataset[i]['text'])
            #     print(described_objects)
            #     print(pred_objects
            #           )
            #     print("--------------")
            #     pred_arr = [objects.reshape(76), predictions[i].numpy().reshape(76)]
            #     binary_array_to_graph(pred_arr, title=valid_dataset[i]['text'], rounded=False)
            # print("worst 5")
            # for i in ind_smallest:#range(len(predictions)):
            #         number_objects = int(len(predictions[0]) / 19)
            #         objects =[0]*len(predictions[0])
            #         for l,obj in enumerate(valid_dataset[i]['catgy']):
            #             for index in obj:
            #                 objects[l*19+index]=1
            #         objects = np.array(objects).reshape(number_objects, 19)
            #         pred_objects_arr = predictions[i].reshape(number_objects, 19)
            #
            #         epsilon = 0.4
            #         described_objects = [[a for a, b in zip(things, objects[i]) if b] for i in range(number_objects)]
            #         pred_objects = [[a for a, b in zip(things, pred_objects_arr[i]) if b>epsilon] for i in range(number_objects)]
            #         print(f"ap: {average_precision_arr[i]}")
            #         print("--------------")
            #         print(valid_dataset[i]['text'])
            #         print(described_objects)
            #         print(pred_objects
            #               )
            #         print("--------------")
            #         pred_arr = [objects.reshape(76),predictions[i].numpy().reshape(76)]
            #
            #         binary_array_to_graph(pred_arr, title=valid_dataset[i]['text'],rounded=False)
    #
    #     print()

print(pred_ratio_list)