from pickletools import optimize
from signal import default_int_handler
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from loguru import logger
import torch.nn.functional as F
import network
import loss
import itertools
from model_loader import load_model
from evaluate import mean_average_precision
from torch.nn import Parameter
from torch.autograd import Variable
from memoryBank import MemoeyBank
from DomainDiscriminator import DomainDiscriminator

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

# Kmeans
def kmeans(x, ncluster, niter=10):
    '''
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] 
    ncluster = min(ncluster, N)
    for i in range(niter):
        # assign all pixels to the closest codebook element
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # move each codebook element to be the mean of the pixels that assigned to it
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        # print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    return c


def train(
          train_s_dataloader,
          train_t_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          evaluate_interval,
          source,
          target,
          method,
          tag,
          lamda = 1,
          t = 0.5
          ):
    # Vgg16
    model = load_model(arch, code_length,num_class)
    model.to(device)
    model.train()

    domain_discri = DomainDiscriminator(in_feature=4096, hidden_size=1024).to(device)
    best_mAP = 0
    all_mAP = 0
    warmup_epoch = 1

    optimizer_warmup = optim.SGD(model.get_Other_parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.SGD(model.get_Other_parameters() + domain_discri.get_parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)

    start = time()
    criterion_CE = loss.OrthoHashLoss() 
    criterion_CLS = loss.OrthoHashLoss() 
    criterion_ADV = loss.DomainAdversarialLoss(domain_discri) 


    Ms = MemoeyBank(1000, 4096, device)  
    Mt = MemoeyBank(1000, 4096, device)  

    for epoch in range(max_iter):
        
        if epoch <= warmup_epoch:
            for (data_s, _, target_s,index) in train_s_dataloader:
                bs = data_s.shape[0]
                data_s = data_s.to(device)
                target_s = target_s.to(device)
                optimizer_warmup.zero_grad()
                target_s_no_onehot = torch.argmax(target_s,dim=1)

                
                logit_s, f_s, code_s = model(data_s)
                cluster_center = kmeans(f_s, ncluster=num_class, niter=20) # (num-class,f-dim)
                kmeans_logits = f_s @ cluster_center.T 
                kmeans_logits = torch.argmax(torch.softmax(kmeans_logits,dim=1),dim=1) 
                mask_s_clean = torch.tensor([False] * bs)

                for i in range(bs):
                    idx = target_s_no_onehot[i] 
                    cnt = torch.zeros(num_class)
                    for j in range(bs):
                        if target_s_no_onehot[j] == idx:
                            cnt[kmeans_logits[j]] += 1
                    if kmeans_logits[i] == torch.argmax(cnt):
                        mask_s_clean[i] = True
                Ms.enqueue_dequeue(f_s[mask_s_clean].detach(), target_s_no_onehot[mask_s_clean])

                loss_ce, _ = criterion_CE(logit_s, code_s, target_s) 
                loss_ce.backward()
                optimizer_warmup.step()

            logger.info('[Epoch:{}/{}][loss_all:{:.4f}]'.format(epoch+1, max_iter, (loss_ce).item()))
            # Evaluate
            if (epoch % evaluate_interval == evaluate_interval-1):
                mAP = evaluate(model,
                                query_dataloader,
                                retrieval_dataloader,
                                code_length,
                                device,
                                topk,
                                tag,
                                source,
                                target
                                )
                all_mAP += mAP
                best_mAP = max(best_mAP, mAP)
                logger.info('[iter:{}/{}][mAP:{:.4f}]'.format(
                    epoch+1,
                    max_iter,
                    mAP,
                ))
        else:
            for ((data_s, data_s_aug, target_s, _), (data_t, data_t_aug, target_t_gt, _)) in zip(train_s_dataloader, train_t_dataloader):
                bs = data_s.shape[0]
                optimizer.zero_grad()
                data_s = data_s.to(device)
                target_s = target_s.to(device)
                target_s_no_onehot = torch.argmax(target_s,dim=1)
                data_t = data_t.to(device)
                logit_s, f_s, code_s = model(data_s)
                logit_t, f_t, code_t = model(data_t)

                
                class_center = torch.zeros(num_class, f_s.shape[1]).to(device)  #(num_class, f_dim)
                class_count = torch.zeros(num_class).to(device)
                Ms_feature, Ms_target = Ms.get() 

                for idx in range(Ms_target.shape[0]):
                    label = Ms_target[idx]
                    class_count[label.type(torch.long)] += 1 
                    class_center[label.type(torch.long)] += Ms_feature[idx,:]
                
                for idx in range(num_class):
                    if class_count[idx] != 0:
                        class_center[idx] /= class_count[idx]
                
                p_s = f_s @ class_center.T
                p_s = torch.softmax(p_s,dim=1) # (bs, num_class)
                mask_s_clean = torch.tensor([False] * f_s.shape[0])
            
                for idx in range(f_s.shape[0]):
                    if p_s[idx,target_s_no_onehot[idx]] > t:
                        mask_s_clean[idx] = True
                Ms.enqueue_dequeue(f_s[mask_s_clean,:].detach(), target_s_no_onehot[mask_s_clean])

                p_t = f_t @ class_center.T
                p_t = torch.softmax(p_t,dim=1) # (bs, num_class)
                mask_t_clean = torch.tensor([False] * f_t.shape[0])
                target_t_no_onehot = torch.zeros(f_t.shape[0])
                label_t = torch.argmax(p_t,dim = 1) 
                for idx in range(f_t.shape[0]):
                    target_t_no_onehot[idx] = label_t[idx]
                    if p_t[idx,label_t[idx]] > t:
                        mask_t_clean[idx] = True

                Mt.enqueue_dequeue(f_t[mask_t_clean,:].detach(), target_t_no_onehot[mask_t_clean])

                loss_cls,_ = criterion_CLS(logit_s[mask_s_clean], code_s[mask_s_clean], target_s_no_onehot[mask_s_clean], False)
                loss_adv = criterion_ADV(f_s[mask_s_clean], f_t[mask_t_clean])
                loss_all = loss_cls + loss_adv

                loss_all.backward()
                optimizer.step()
                
            logger.info('[Epoch:{}/{}][loss_all:{:.4f}]'.format(epoch+1, max_iter, loss_all.item()))
            # Evaluate
            if (epoch % evaluate_interval == evaluate_interval-1):
                mAP = evaluate(model,
                                query_dataloader,
                                retrieval_dataloader,
                                code_length,
                                device,
                                topk,
                                tag,
                                source,
                                target
                                )
                all_mAP += mAP
                best_mAP = max(best_mAP, mAP)
                logger.info('[iter:{}/{}][mAP:{:.4f}]'.format(
                    epoch+1,
                    max_iter,
                    mAP,
                ))

    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   tag,
                   source,
                   target
                   )
    
    logger.info('Training finish, [iteration:{}][mAP:{:.4f}]'.format(epoch+1, mAP))
    logger.info('Training finish, Average mAP:{:.4f}, Best mAP:{:.4f}'.format((all_mAP * evaluate_interval)/ max_iter, best_mAP))
    end = time()
    logger.info('Training Finish, time:{}'.format((end - start) / 60))





def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, tag, source, target):
    model.eval()
    source = source.split('/')[-1].split('.')[0]
    target = target.split('/')[-1].split('.')[0]
    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    
    # One-hot encode targets

    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
   
    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    # np.save("hashcode/{}/query_code_{}_{}_{}_mAP_{}".format(tag, source, target, code_length, mAP), query_code.cpu().detach().numpy())
    # np.save("hashcode/{}/retrieval_code_{}_{}_{}_mAP_{}".format(tag,source, target, code_length, mAP), retrieval_code.cpu().detach().numpy())
    # np.save("hashcode/{}/query_target_{}_{}_{}_mAP_{}".format(tag,source, target, code_length, mAP), onehot_query_targets.cpu().detach().numpy())
    # np.save("hashcode/{}/retrieval_target_{}_{}_{}_mAP_{}".format(tag, source, target, code_length, mAP), onehot_retrieval_targets.cpu().detach().numpy())

    model.train()
    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length],dtype=torch.float)
        for data, _,_,index in dataloader:
            data = data.to(device)
            _,_,outputs= model(data)
            code[index, :] = outputs.sign().cpu() 

    return code








    


    




