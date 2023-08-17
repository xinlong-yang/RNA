import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import random
from torch.autograd import Function

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, batch_size, device, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1]
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1))) 

    if batch_size == ad_out.size(0) // 2 :
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    else:
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * (ad_out.size(0) - batch_size))).float().to(device)
    
    # 
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()

    return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net,batch_size, device):
    ad_out = ad_net(features)
    if batch_size == ad_out.size(0) // 2 :
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    else:
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * (ad_out.size(0) - batch_size))).float().to(device)
    return nn.BCELoss()(ad_out, dc_target)


def wasserstein1d(x, y, aggregate=True):
    """Compute wasserstein loss in 1D"""
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    n = x.size(0)
    # need scalar
    if aggregate:
        z = (x1-y1).view(-1)
        return torch.dot(z, z)/n
    else:
        return (x1-y1).square().sum(0) / n


def quantization_swdc_loss(b, device='cuda', aggregate=True):
    
    real_b = torch.randn(b.shape, device=device).sign()
    bsize, dim = b.size() 
    if aggregate:
        gloss = wasserstein1d(real_b, b) / dim
    else:
        gloss = wasserstein1d(real_b, b, aggregate=False)

    return gloss

class BaseClassificationLoss(nn.Module):
    def __init__(self):
        super(BaseClassificationLoss, self).__init__()
        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        raise NotImplementedError

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc

# 这一部分和One-Loss一样
class OrthoHashLoss(BaseClassificationLoss):

    def __init__(self,
                 ce=1,
                 s=8,
                 m=0.2,
                 m_type='cos',  # cos/arc
                 multiclass=False,
                 quan=0,
                 quan_type='cs', 
                 multiclass_loss='label_smoothing',
                 **kwargs):
        super(OrthoHashLoss, self).__init__()
        self.ce = ce
        self.s = s
        self.m = m
        self.m_type = m_type
        self.multiclass = multiclass

        self.quan = quan
        self.quan_type = quan_type
        self.multiclass_loss = multiclass_loss
        assert multiclass_loss in ['bce', 'imbalance', 'label_smoothing']

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, logits, code_logits, labels, onehot=True):
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            margin_logits = self.compute_margin_logits(logits, labels)

            if self.multiclass_loss in ['bce', 'imbalance']:
                loss_ce = F.binary_cross_entropy_with_logits(margin_logits, labels, reduction='none')
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels,
                                                                        labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()
            elif self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce = loss_ce.mean()
            else:
                raise NotImplementedError('unknown method: {self.multiclass_loss}')
        else:
            if onehot:
                labels = labels.argmax(1)
            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)
            loss_ce_batch = F.cross_entropy(margin_logits, labels, reduction='none')


        if self.quan != 0:
            if self.quan_type == 'cs':
                quantization = (1. - F.cosine_similarity(code_logits, code_logits.detach().sign(), dim=1))
            elif self.quan_type == 'l1':
                quantization = torch.abs(code_logits - code_logits.detach().sign())
            else:  # l2
                quantization = torch.pow(code_logits - code_logits.detach().sign(), 2)
            quantization_batch = quantization
            quantization = quantization.mean()
        else:
            quantization_batch = torch.zeros_like(loss_ce_batch)
            quantization = torch.tensor(0.).to(logits.device)

       
        self.losses['ce'] = loss_ce
        self.losses['quan'] = quantization
        loss = self.ce * loss_ce + self.quan * quantization
        loss_batch = self.ce * loss_ce_batch + self.quan * quantization_batch
        return loss, loss_batch



class DAPH_LOSS(BaseClassificationLoss):
    def __init__(self, quan = 0.001, align = 0.1):
        super(DAPH_LOSS,self).__init__()
        self.w_quan = quan
        self.w_align = align
    

    def forward(self,feature_s,feature_t,code_s,code_t):
        center_s = torch.sum(feature_s,dim=0) / feature_s.shape[0]
        center_t = torch.sum(feature_t,dim=0) / feature_t.shape[0]
        loss_align = torch.pow((center_s - center_t), 2).sum() / center_s.shape[0]

        loss_quantization = torch.pow(code_s - code_s.detach().sign(),2).sum() / code_s.shape[0] \
                       +  torch.pow(code_t - code_t.detach().sign(),2).sum() / code_t.shape[0]
        # print(self.w_quan * loss_quantization , self.w_align * loss_align)
        return self.w_quan * loss_quantization + self.w_align * loss_align



class DHLing_LOSS(BaseClassificationLoss):
    def __init__(self, quan = 0.001, cc = 0.1, m = 0.1):
        
        super(DHLing_LOSS,self).__init__()
        self.w_quan = quan
        self.w_cc = cc
        self.w_m = m
    

    def forward(self,model,class_center,target_s,code_s,code_t):
        
        loss_m = torch.pow(model.Wg.weight - model.Ds,2).mean()  + torch.pow(model.Wg.weight - model.Dt,2).mean() 

        # pairwise similarity preserving
        target_t = code_t @ class_center.T
        
        idx_target_t = torch.argmax(target_t,dim=1)
        idx_target_s = torch.argmax(target_s,dim=1)
        idx_target_s.unsqueeze_(dim=0)
        idx_target_t.unsqueeze_(dim=0)
        
        inner_dot = 0.5 * (code_s.sign() @ code_t.sign().T) / (code_s.shape[0] * code_t.shape[0]) 
        p = 1 / (1 + torch.exp(-inner_dot))
        mask1 = (idx_target_s == idx_target_t.T).T
        mask2 = (idx_target_s != idx_target_t.T).T
        
        loss_cc = (-torch.log(p) * mask1 + -torch.log(1 - p) * mask2).sum() / (mask1.shape[0] * mask2.shape[0])

        # quantization
        loss_quantization = torch.pow(code_s - code_s.detach().sign(),2).sum() / code_s.shape[0] \
                       +  torch.pow(code_t - code_t.detach().sign(),2).sum() / code_t.shape[0]

        # print(self.w_quan * loss_quantization , self.w_cc * loss_cc , self.w_m * loss_m)
        return self.w_quan * loss_quantization + self.w_cc * loss_cc + self.w_m * loss_m


class CSQLoss(torch.nn.Module):
    def __init__(self, nclass,bit, device):
        super(CSQLoss, self).__init__()
        self.is_single_label = True
        self.hash_targets = self.get_hash_targets(nclass,bit).to(device)
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(device)
        self.criterion = torch.nn.BCELoss().to(device)

    def forward(self, u, y):
        
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss +  0.0001 * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets



class DomainAdversarialLoss(nn.Module):
    r"""
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is
    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].
    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.
    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.
    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.
    Examples::
        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction = 'mean',
                 grl = None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s = None, w_t = None):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.narrow(0,0,f_s.size(0)), d.narrow(0, f_s.size(0), f.size(0) - f_s.size(0))
            d_label_s = torch.ones((f_s.size(0), 1),dtype=torch.float).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1),dtype=torch.float).to(f_t.device)
            # print('------------------------------------------------------')
            # print(d_s.shape, d_t.shape, d_label_s.shape, d_label_t.shape)
            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            # print(d_s.dtype, d_t.dtype, d_label_s.dtype, d_label_t.dtype,w_s.dtype, w_t.dtype )
            return 0.5 * (
                
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)



class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) :
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start
        The forward and backward behaviours are:
        .. math::
            \mathcal{R}(x) = x,
            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.
        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:
        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo
        where :math:`i` is the iteration step.
        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha = 1.0, lo = 0.0, hi = 1.,
                 max_iters = 1000., auto_step = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

