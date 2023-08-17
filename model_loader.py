import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.nn import Parameter

class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

         # if no centroids, by default just usual weight
        codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
       
        if not learn_cent:
           
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

def load_model(arch, code_length, num_cluster=30):
    """
        Load CNN model.
    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    
    if arch == 'alexnet':
       
        model = models.alexnet(pretrained=True)
        # model.classifier = model.classifier[:-2]
        model = ModelWrapper(model, code_length, num_cluster)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        # model.classifier = model.classifier[:-3]
        model = ModelWrapper(model, code_length, num_cluster)
    else:
        raise ValueError("Invalid model name!")

    return model


class ModelWrapper(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, code_length, num_cluster):
        super(ModelWrapper, self).__init__()
        
        self.model = model
        self.code_length = code_length
        self.class_num = num_cluster
        
        self.Wf = self.model.classifier[0]
        self.Wg = nn.Linear(self.model.classifier[3].in_features, self.model.classifier[3].out_features, bias=False)
        self.Wh = nn.Linear(self.model.classifier[6].in_features, self.code_length, bias=False)
        
        self.feature = self.model.features
        self.avgpool = self.model.avgpool

      
        self.Ds = nn.Parameter(torch.randn(self.Wg.in_features, self.Wg.out_features),requires_grad = True)
        self.alphaS = nn.Parameter(torch.randn(1, self.Wg.in_features),requires_grad=True)
        self.Dt = nn.Parameter(torch.randn(self.Wg.in_features, self.Wg.out_features), requires_grad=True)
        self.alphaT = nn.Parameter(torch.randn(1, self.Wg.in_features),requires_grad=True)

        
        self.getF = nn.Sequential(
            nn.Dropout(p = 0.5, inplace=False),
            self.Wf,
            nn.ReLU(inplace=False)
        )
        
        self.getG = nn.Sequential(
            nn.Dropout(p = 0.5, inplace=False),
            self.Wg,
            nn.ReLU(inplace=False)
        )
        
        self.getH = nn.Sequential(
            self.Wh ,
            nn.Tanh()
        )
        # Extract features
        self.extract_features = False
       
        self.ce_fc = nn.Linear(code_length, num_cluster, bias=False)
    
    def forward(self, x):
       
        if self.extract_features:
            x = self.feature(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            f = self.getF(x)
            return f

        else:
          
            x = self.feature(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
           
            f = self.getF(x)
            g = self.getG(f)
            h = self.getH(g) 
            logits = self.ce_fc(h)
            return logits,f,h

    def get_Other_parameters(self):
        
        parameter_list = [  {"params":self.feature.parameters(), "lr_mult":0.1},
                            {"params":self.getF.parameters(), "lr_mult":1},
                            {"params":self.getG.parameters(), "lr_mult":1},
                            {"params":self.getH.parameters(), "lr_mult":1},
                            {"params":self.ce_fc.parameters(), "lr_mult":1}                        
                        ]
        return parameter_list

    def get_dict_parameters(self):
        parameter_list = [  
                            {"params":self.Ds, "lr_mult":1},
                            {"params":self.Dt, "lr_mult":1},
                            {"params":self.alphaS, "lr_mult":1},
                            {"params":self.alphaT, "lr_mult":1}                       
                        ]
        return parameter_list

    def freezenDic(self,f):
        
        self.Ds.requires_grad_(f)
        self.Dt.requires_grad_(f)
        self.alphaS.requires_grad_(f)
        self.alphaT.requires_grad_(f)

    def freezenOther(self,f):
        
        for param in self.feature.parameters():
            param.requires_grad_(f)
        for param in self.getF.parameters():
            param.requires_grad_(f)
        for param in self.getG.parameters():
            param.requires_grad_(f)
        for param in self.getH.parameters():
            param.requires_grad_(f)
        for param in self.ce_fc.parameters():
            param.requires_grad_(f)

    def set_extract_features(self, flag):
        """
        Extract features.
        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag


