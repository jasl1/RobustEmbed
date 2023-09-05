import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead


def embed_encode(encoder, input_ids=None, token_type_ids=None):
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    #print('embed_encode 292 : ',input_ids.shape, token_type_ids.shape)
    embedding_output = encoder.bert.embeddings(input_ids, token_type_ids)
    return embedding_output


def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
    return loss


def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
    ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
    if reduce:
        return (p * (rp - ry) * 2).sum() / bs
    else:
        return (p * (rp - ry) * 2).sum()
        

#############################################
k_step_PGD = 1 #3***
_stable_kl_ = True #True, False **
_model_name_for_PGD_perturbation_ = 'bert'

#adversarial training
max_iters2=2
_epsilon_=1e-5 # 1e-6**
_norm_p_="inf"  #"inf", "l2", "l1"
_norm_level_=0 #0 ***, 1


max_iters1=2
alpha = 0.00097


_noise_var_=1e-5 # 1e-5 **
_step_size_=1e-5 #1e-5 ***, 1e-4, 1e-3, 1e-2
perturb_delta = 0.5
##############################################
    
class GenerateMLMbasedPerturbation:
    def __init__(
        self,
        epsilon=1e-6,
        step_size=1e-3,
        noise_var=1e-5,
        norm_p="inf",
        k_PGD=1,
        perturb_delta = 0.5,
        max_iters1=5,
        max_iters2=5,
        alpha = 0.00627, # 0.00314
        #encoder_type=EncoderModelType.BERT,
        norm_level=0,
    ):
        super(GenerateMLMbasedPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.k_PGD = k_PGD
        self.perturb_delta = perturb_delta
        self.max_iters1 = max_iters1
        self.max_iters2 = max_iters2
        self.alpha = alpha
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        #self.encoder_type = encoder_type
        self.sentence_level = norm_level
        
    def norm_grad(self, grad, eff_grad=None, sentence_level=False):
        eff_direction = None
        if self.norm_p == "l2": #what is self.norm_p??
            if sentence_level:
                direction = grad / (
                    torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon #what is self.epsilon??
                )
            else:
                direction = grad / (
                    torch.norm(grad, dim=-1, keepdim=True) + self.epsilon
                )
        elif self.norm_p == "l1":
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (
                    grad.abs().amax((-2, -1), keepdim=True)[0] + self.epsilon
                )
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (
                    grad.abs().max(-1, keepdim=True)[0] + self.epsilon
                )
        return direction, eff_direction
        
    def generate_noise(self, embed, mask, noise_var=1e-5):
        noise = embed.data.new(embed.size()).normal_(0, 1) * noise_var
        noise.detach()
        noise.requires_grad_()
        return noise
       
    def calculate_PGD_perturbation(self,
        encoder,
        model_name,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None):
        
        #embedding = embed_encode(encoder, input_ids, token_type_ids)
        #noise = self.generate_noise(embedding, attention_mask, noise_var=self.noise_var)
        #print("noise: ", noise.size())

        #encoder.eval()
        
        '''
        outputs = encoder(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
           
        logits = outputs[0]
        if logits.requires_grad == False:
            return outputs
        
          
        for step in range(self.k_PGD):
        
            outputs = encoder(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=embedding + noise, #embed.data.detach() or/and noise.detach() ??
                )
            
            adv_logits = outputs[0]
            
            #print('requires_grad1: ', adv_logits.requires_grad, logits.requires_grad)
            
            #print('adv_logits:', len(outputs), outputs, type(adv_logits), adv_logits)
                
            if _stable_kl_ == True:
                if model_name == 'bert':
                    adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
                else:
                    adv_loss = stable_kl(adv_logits, logits.detach())
            else:
                adv_loss = KL(adv_logits, logits.detach(), reduction="batchmean")
            
            #print('requires_grad2: ', adv_loss.requires_grad)
            
            (delta_grad,) = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
    
            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
        
            eff_delta_grad = delta_grad * self.step_size #what is self.step_size??
            delta_grad = noise + delta_grad * self.step_size
    
            noise, eff_noise = self.norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.sentence_level)
            
            noise = noise.detach()
            noise.requires_grad_()
            #print('pooooooooooooooooonnnnnnnnnnnnnneeeeeeeeeeeeehhhhhhhhhhhhhh11: ')
        
        '''
        '''
        if labels is not None:
                max_iters = max(self.max_iters1, self.max_iters2)
                for _iter in range(max_iters):
                    outputs = encoder(
                        None,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        inputs_embeds=embedding + noise, #embed.data.detach() or/and noise.detach() ??
                        labels=labels
                        )
                    
                    #loss, adv_logits = outputs[:2] # print loss???
                    loss = outputs[0]
                    #print("\n\noutputs: ",outputs, type(outputs))
                    #print("\loss: ",loss)
                    
                    (delta_grad,) = torch.autograd.grad(loss, noise, only_inputs=True, retain_graph=False)
            
                    norm = delta_grad.norm()
                    if torch.isnan(norm) or torch.isinf(norm):
                        return outputs
                
                    scaled_g = torch.sign(delta_grad.data)
                    
                    eff_delta_grad = delta_grad * self.step_size 
                    delta_grad = noise + delta_grad * self.step_size
                    
                           
                            
                    if _iter < self.max_iters1:
                        if _iter < self.max_iters2:
                            noise.data = self.perturb_delta*(noise.data + self.alpha * scaled_g)
                        else:
                            noise.data += self.alpha * scaled_g
                                
                    if _iter < self.max_iters2:
                        noise_x, eff_noise = self.norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.sentence_level)
                        if _iter < self.max_iters1:
                            noise.data += (1-self.perturb_delta)*noise_x.data
                        else:
                            noise.data = noise_x.data
                    
                    
                    """eff_delta_grad = delta_grad * self.step_size #what is self.step_size??
                    delta_grad = noise + delta_grad * self.step_size
            
                    noise, eff_noise = self.norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.sentence_level)"""
                    
                    noise = noise.detach()
                    noise.requires_grad_()
                    #print('pooooooooooooooooonnnnnnnnnnnnnneeeeeeeeeeeeehhhhhhhhhhhhhh11: ')
        
        
                  
        encoder.train()           
        if labels is not None:                    
            outputs = encoder(
                    #input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    inputs_embeds=embedding.data.detach() + noise.detach(), # noise.detach() ??
                    )
        else:
            outputs = encoder(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    #inputs_embeds=embedding.data.detach(), # noise.detach() ??
                    )       
        '''
        
        outputs = encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                    #inputs_embeds=embedding.data.detach(), # noise.detach() ??
                )      
                    
        
        '''logits = outputs[0]  
           
        if _stable_kl_ == True:
            adv_loss_f = stable_kl(adv_logits, logits.detach(), reduce=False)
            adv_loss_b = stable_kl(logits, adv_logits.detach(), reduce=False)
        else: 
            adv_loss_f = KL(adv_logits, logits.detach())
            adv_loss_b = KL(logits, adv_logits.detach())
            
        adv_loss = (adv_loss_f + adv_loss_b)'''
        
        return outputs
        

#AutoModelForSequenceClassification
#BertPreTrainedModel
class BertForAT(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        #print(type(config), config._name_or_path)
        self.adv_training = GenerateMLMbasedPerturbation(epsilon=_epsilon_, step_size=_step_size_, noise_var=_noise_var_, norm_p=_norm_p_, k_PGD=k_step_PGD, perturb_delta = perturb_delta, max_iters1=max_iters1, max_iters2=max_iters2, alpha=alpha, norm_level=_norm_level_)
            
        self.bert = transformers.BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=config._name_or_path,config=config)

        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None
    ):
        #print("\nlabels: ",labels, type(labels))
        outputs = self.adv_training.calculate_PGD_perturbation(self.bert,
                        _model_name_for_PGD_perturbation_,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
        #print('outputs:', outputs[0])
        return outputs

