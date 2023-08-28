import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)
        
        
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = ProjectionMLP(config) if model_args_batchnorm else MLPLayer(config) 
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


###############################
regularize = 'other'
random_start = True
_norm_level_2=1 #0 ***, 1

_norm_p_2="inf"  #"inf", "l2", "l1"


_epsilon2_=1e-5 # 1e-6**
step_size_2=1e-5 #1e-5 ***, 1e-4, 1e-3, 1e-2
###############################

epsilon=1e-6
noise_var=1e-5

max_iters1=5
max_iters2=5


alpha = 0.00097  # 0.00314, 0.00627

perturb_delta = 0.5

cl_adv_weight = 128 #Table 11 --> 32, 64, 128, 256, 512

model_args_batchnorm = False #--> change line 407




mlm_weight = 0.01

Sim_SCE = False
USCAL = True
# Sim_SCE --> Sim_SCE = True & comment first part & comment second part
# USCAL -->  Sim_SCE = True & USCAL = True & comment second part
# our method --> Sim_SCE = False & USCAL = True

#lines -->  line 321 for output return, lines 713 --> grads for adv_cl
#################################

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    model_name = 'bert',
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)


    #print("batch_size: " + str(batch_size))
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    
    #adv_loss = 0.0
    ############################################################################ [Sim_SCE = True (only Sim_SCE)] or [remove second part (lambda)]
       
    perturb_generator = GeneratePerturbation(epsilon=epsilon, step_size=step_size_2, norm_level=_norm_level_2, norm_p=_norm_p_2, epsilon2=_epsilon2_, noise_var=noise_var, perturb_delta = perturb_delta, max_iters1=max_iters1, max_iters2=max_iters2, alpha=alpha, weight=cl_adv_weight, regularize=regularize, batch_size = batch_size)
    
    advinputs, adv_loss = perturb_generator.calculate_PGD_perturbation(cls, encoder,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        random_start = random_start)
    
    ##########################################################################################
    
    #embedding = embed_encode(encoder, input_ids, token_type_ids)
    #inputs = torch.cat((embedding, advinputs))
    
    #print('attention_mask :', attention_mask)
    #print('token_type_ids :', token_type_ids)
    
    outputs12, z1, z2 = generate_representation(
                 cls,
                 encoder,
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 token_type_ids=token_type_ids,
                 position_ids=position_ids,
                 head_mask=head_mask,
                 inputs_embeds=inputs_embeds,
                 output_attentions=output_attentions,
                 output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                 return_dict=True,
                 batch_size = batch_size,
                 num_sent = 2)
                 
    if Sim_SCE == False:           
        outputs3, z3 = generate_representation(
                cls,
                encoder,
                input_ids=None,
                attention_mask=attention_mask[:batch_size,:],
                token_type_ids=token_type_ids[:batch_size,:] if token_type_ids is not None else token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=advinputs,
                output_attentions=output_attentions,
                output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                batch_size = batch_size,
                num_sent = 1)
    
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)


    if Sim_SCE == False:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 0)
        labels13 = torch.arange(z1_z3_cos.size(0)).long().to(cls.device)
        labels = torch.cat([labels, labels13], 0)
    
    loss_fct = nn.CrossEntropyLoss()
    simloss = loss_fct(cos_sim, labels)
    
    #print('simloss: ', simloss, adv_loss) 
    total_loss = simloss + adv_loss
    #total_loss = simloss
        
    #print('simsce: ', simloss, type(simloss), '\n')
    #print('cl_adv_loss: ', adv_loss, type(adv_loss), '\n') 
     
    # MLM auxiliary objective
    mlm_outputs = None
    ###############################################################################################################
    '''
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
    '''  
    ################################################################################################################
        
    # Calculate loss for MLM
    logits = None
    
    '''
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        logits = prediction_scores.view(-1, cls.config.vocab_size) 
        
        ##########################################################################################
        
        masked_lm_loss = loss_fct(logits, mlm_labels.view(-1))
        #print('masked_lm_loss: ', masked_lm_loss) 
        total_loss += mlm_weight * masked_lm_loss
        #print('masked_lm_loss: ', masked_lm_loss, type(masked_lm_loss), '\n')
        
        ##########################################################################################
    '''
    
    #line 321
    # outputs12 = cos_sim, mlm_outputs = logits 
    if not return_dict:
        output = (cos_sim,) + outputs12[2:]#z1_z2_cos
        return ((total_loss,) + output) if total_loss is not None else output
    return SequenceClassifierOutput(
        loss=total_loss,
        logits=cos_sim,
        hidden_states=outputs12.hidden_states,
        attentions=outputs12.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


def embed_encode(encoder, input_ids=None, token_type_ids=None):
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    #print('embed_encode 292 : ',input_ids.shape, token_type_ids.shape)
    embedding_output = encoder.embeddings(input_ids, token_type_ids)
    return embedding_output
    

def generate_representation(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    batch_size = 64,
    num_sent = 2,
):

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds, #embed.data.detach() should be tested with input_ids
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
   
   
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    if cls.pooler_type == "cls":
        #pooler_output = cls.mlp(pooler_output)
        
        #--> change line 407
        
        pooler_output = pooler_output.view((batch_size*num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
        pooler_output = cls.mlp(pooler_output)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
        
           
    #print('pooler_output after mlp:', pooler_output.shape)
    
    if num_sent == 2:
        z1, z2 = pooler_output[:,0], pooler_output[:,1]
        #print('pooler_output z1:', pooler_output.shape, z1.shape)
    else:
        z3 = pooler_output[:, 0]   
        #print('pooler_output z3:', pooler_output.shape, z3.shape)
        
    #lines 432
    if dist.is_initialized(): #and cls.training???
        if num_sent == 1:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)
        else:    
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]

            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2

            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
        
    if num_sent == 1:   
        return outputs, z3
    else:
        return outputs, z1, z2
        
                               
class GeneratePerturbation:
    def __init__(
        self,
        epsilon=1e-6,
        step_size=1e-3,
        norm_level=0,
        norm_p="inf",
        epsilon2=1e-6,
        noise_var=1e-5,
        perturb_delta = 0.5,
        max_iters1=5,
        max_iters2=5,
        alpha = 0.00627, # 0.00314
        weight = 256, #Table 11 --> 32, 64, 128, 512
        regularize = 'original',
        batch_size = 64,
        ):
            super(GeneratePerturbation, self).__init__()
            self.epsilon = epsilon
            self.step_size = step_size
            self.sentence_level = norm_level
            self.norm_p = norm_p
            self.epsilon2 = epsilon2
            self.noise_var = noise_var
            self.perturb_delta = perturb_delta
            self.max_iters1 = max_iters1
            self.max_iters2 = max_iters2
            self.alpha = alpha
            self.weight = weight
            self.regularize = regularize
            self.batch_size = batch_size   
    
    def norm_grad(self, grad, eff_grad=None, sentence_level=False):
        eff_direction = None
        if self.norm_p == "l2": #what is self.norm_p??
            if sentence_level:
                direction = grad / (
                    torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon2 
                )
            else:
                direction = grad / (
                    torch.norm(grad, dim=-1, keepdim=True) + self.epsilon2
                )
        elif self.norm_p == "l1":
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (
                    grad.abs().amax((-2, -1), keepdim=True)[0] + self.epsilon2
                )
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon2)
                eff_direction = eff_grad / (
                    grad.abs().max(-1, keepdim=True)[0] + self.epsilon2
                )
        return direction, eff_direction
        
    def generate_noise(self, embed, mask, noise_var=1e-5):
        noise = embed.data.new(embed.size()).normal_(0, 1) * noise_var
        #noise = torch.FloatTensor(embed.shape).uniform_(-noise_var, noise_var)
        noise = noise.float().cuda()
        #noise.detach()
        #noise.requires_grad = False
        return noise
        
    def project(self, x, original_x, epsilon):
        max_x = original_x + epsilon
        min_x = original_x - epsilon
        x = torch.max(torch.min(x, max_x), min_x)
        return x
            
    def calculate_PGD_perturbation(self, cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        random_start = True):
        
            #print('line 576 : ',input_ids.shape, token_type_ids.shape)
            embedding = embed_encode(encoder, input_ids, token_type_ids)
            #z = y.detach().clone().requires_grad_(True)
            x = embedding[:self.batch_size,:].detach().float().clone()
            target =  embedding[self.batch_size:,:].float().clone() 
            
            if random_start:       
                noise = self.generate_noise(x, attention_mask[:self.batch_size,:], noise_var=self.noise_var)
                x += noise
            
            x.requires_grad = True
            
            
            encoder.eval() #lines 670
            cls.pooler.eval()   
            cls.mlp.eval()  
            
            max_iters = max(self.max_iters1, self.max_iters2)
            with torch.enable_grad():
                loss_fct = nn.CrossEntropyLoss()
                for _iter in range(max_iters):
                    encoder.zero_grad() #or cls.eval() 
                    cls.pooler.zero_grad()
                    cls.mlp.zero_grad()
                    
                    inputs = torch.cat((x, target))
                    
                    output, z1, z2 = generate_representation(
                        cls,
                        encoder,
                        input_ids=None,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs,
                        output_attentions=output_attentions,
                        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                        return_dict=True,
                        batch_size = self.batch_size,
                        num_sent = 2)
 
                    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
                    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
                    loss = loss_fct(cos_sim, labels)
                   
                    delta_grad = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]
                    
                    #lines 713
                    ########################################################### method 1
                    
                    scaled_g = torch.sign(delta_grad.data) #lines 713
                    #x.data += self.alpha * scaled_g #lines 713
                    
                    ###########################################################
                    
                    norm = delta_grad.norm()
                    if torch.isnan(norm) or torch.isinf(norm):
                        return x.detach(), loss
                    
                    
                    ##################### method 2
                     
                    #x, eff_x = self.norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.sentence_level) #what is self.sentence_level??
                    
                    ##################### method 3
                    
                    
                    eff_delta_grad = delta_grad * self.step_size #what is self.step_size??
                    delta_grad = x + delta_grad * self.step_size
                    
                    if _iter < self.max_iters1:
                        if _iter < self.max_iters2:
                            x.data = self.perturb_delta*(x.data + self.alpha * scaled_g)
                        else:
                            x.data += self.alpha * scaled_g
                        
                    if _iter < self.max_iters2:
                        
                        xx, eff_x = self.norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.sentence_level)
                        if _iter < self.max_iters1:
                            x.data += (1-self.perturb_delta)*xx.data
                        else:
                            x.data = xx.data
                    #####################
                    
                    x = x.detach()
                    x.requires_grad_()
                   
                    x = self.project(x, embedding[:self.batch_size,:], self.epsilon)
          
                  
            encoder.train()
            cls.pooler.train()
            cls.mlp.train()  
            
            if self.regularize== 'original':
                if USCAL == False:
                    inputs = torch.cat((x, embedding[:self.batch_size,:]))
                else:
                    inputs = torch.cat((embedding[:self.batch_size,:], x))
            else:
                if USCAL == False:
                    inputs = torch.cat((x, target))
                else:
                    inputs = torch.cat((target, x))
                
            output, z1, z2 = generate_representation(
                         cls,
                         encoder,
                         input_ids=None,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         position_ids=position_ids,
                         head_mask=head_mask,
                         inputs_embeds=inputs,
                         output_attentions=output_attentions,
                         output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                         return_dict=True,
                         batch_size = self.batch_size,
                         num_sent = 2)
                        
            cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
            loss = (1/self.weight) * loss_fct(cos_sim, labels)
             
            return x.detach(), loss
               
                        
class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                model_name = 'bert',
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                model_name = 'roberta',
            )
