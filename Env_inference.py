import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="1"
os.environ['HF_EVALUATE_OFFLINE'] = '1'

import sys
import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import re, json
from tqdm import tqdm


from RL.utils import *
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DatasetLoader.collate_func import collate
from DocBuilder.LexMAE import lex_retriever
from DocBuilder.utils import restore_batched_list, generate_mask, tensor_retuen_type
from LM.llama_reader import LLaMa_reader, EncTunedLM
from LM.Knowledge_encoder import KnowEncoder, KnowEncoder_Embedding
from train_ret_2 import NQADataset
from metric.reward import metric
import yaml
import peft

from transformers import AutoTokenizer
import config
import numpy as np


token = "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"
bert_dir = "huggingface/bert"
LM_dir = "/usr/model/llama2-7b/"

if __name__=="__main__":
    '''This code can run final inference to get experement results'''
    print(torch.cuda.device_count())
    device='cuda'
    metric_c = metric()
    metric_c.to(device)
    
    
    print('Loading dataset...')
    num_testing=64
    if False:
        data_path = "data/TV_test.jsonl"
        dataset = NQADataset(data_path=data_path, use_doc=True, use_short=True, use_long=False, num_samples = num_testing+32)
        length = 128
        collate_fn = collate(LM_dir, bert_dir, max_length=length, form="short")
    else:
        data_path = "data/smart_train.jsonl"
        with open(f'data/smart_factory.jsonl','r') as f:
            
            documents=json.load(f)
        dataset = NQADataset(data_path=data_path, use_doc=True, use_short=False, use_long=True, num_samples = num_testing+8, file=documents)
        length = 256
        collate_fn = collate(LM_dir, bert_dir, max_length=length, form = "long")
    dataset = [*dataset]*64
    
    Enc=True
    Policy=False
    print('Loading LLM')
    generate_config = config.generate_config.copy()
    generate_config["temperature"]=1
    if not Policy:
        generate_config["do_sample"]=False
        generate_config["top_k"]=1
    LM = LLaMa_reader(LM_dir, device, token = token, from_pretrained=True, generate_config=generate_config)
    dtype = LM.dtype
    num_dims = LM.model.config.hidden_size
    # print(LM.model.config)
    print(f'Initialize KnowEnc with {dtype}...')
    # Encoder=KnowEncoder(dims = num_dims, **config.enc_config, dtype=dtype)
    Encoder = KnowEncoder_Embedding(embedding = LM.model.get_input_embeddings(), dims = num_dims, **config.enc_config, dtype=dtype)
    
    Encoder.eval()
    print(f'Initialize EncTunedLM...')
    peft_configs = {'Enc': peft.AdaptionPromptConfig(adapter_layers=32, adapter_len=1)}
    LM = EncTunedLM(LM, Enc = Encoder, configs = peft_configs, adapter_name='Enc')
    LM.to(device)

    if Enc and True:
        # torch.save(LM.state_dict(), "/usr/model/EncLM.pt")
        print(f'Loading EncTunedLM weight...')
        LM.load_state_dict(torch.load("save/NQ_EncLM_3.pt", map_location='cpu'), strict= False)
    # init retriever
    LM.eval()

    print('Initilize retriever')
    lex_MAE_retriver=lex_retriever()
    lex_MAE_retriver.to(device)
    lex_MAE_retriver.eval()
    lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=False)
    
    

    max_epoch = 10
    num_retrieve=1
    num_neg=16
    num_RL_update = 8

    env_bs=16
    if Enc:
        env = LLMEnv_test(dataset, LM, lex_MAE_retriver, 3, collate_fn, batch_size=env_bs, shuffle=False, step_size=15 if Policy else 256)
    else:
        env = Orginal_Env(dataset, LM, lex_MAE_retriver, 3, collate_fn, batch_size=env_bs, shuffle=False, step_size=15 if Policy else 256)
    
    print("Initialize Agent...")
    agent = BertAgentCritic(config.agent_size_config, env.action_space_size).to(torch.bfloat16)
    agent.to(device)
    agent.eval()
    if Policy:
        agent.load_state_dict(torch.load("save/TV_Agent_0.pt", map_location="cpu"))
    
    # Training loop
    total = 100000
    memory = []
    ma_reward=0.
    episode=0
    done = [True]*env_bs
    state=[None]*env_bs
    q_list=[]
    a_list=[]
    true_list=[]
    print("Starting reset...")
    f = open("moniter.txt", "a")
    
    for i in range(env_bs):
        if done[i]:
            state[i] = env.reset(i)  # Shape: string
            # env.d_t[i]=[]
            done[i]=False
    while True:
        for i in range(env_bs):
            if done[i]:
                q_list.append(env.x[i])
                a_list.append(env.cat_response(env.response_cache[i], True))
                true_list.append(env.ground_truth[i])
                # print(a_list[-1], "\n", true_list[-1])
                episode+=1
                state[i] = env.reset(i)  # Shape: string
                # env.d_t[i]=[]
                done[i]=False
        if len(q_list)>=num_testing:
            break
        while not any(done):
            with torch.no_grad():
                action_logits, state_value = agent(state)  # token_logits:(B, num, vocab), action_logits shape: (B, action_space_size), state_value shape: (B,)
            action_logits, state_value = action_logits.cpu(), state_value.cpu()
            # action_logits[:,1]-=0.3
            action_dist = Categorical(logits = action_logits/0.1)
            action = action_dist.sample()  # Shape: (B,)
            if not Policy:
                action[:]=1
            next_state, reward, done, _ = env.step(action)  # next_state shape: string, reward shape: scalar, done shape: scalar (boolean)
            print(action[0].item(), end='', flush=True)
            # print(env.cat_response(env.response_cache[0]))
            state = next_state
            
           
    print(a_list[:5], true_list[:5])
    # normalize
    true_list = [t.lower() if isinstance(t, str) else [e.lower() for e in t] for t in true_list]
    maching=None
    if isinstance(true_list[0],list):
        a_list = [re.sub(",\.", "", a.lower().strip()) for a in a_list]
        maching = [a_list[i] in true_list[i] for i in range(len(a_list))]
        true_list = [t[0] for t in true_list]
        
    
    bert = metric_c.Bert_score(a_list, true_list )
    R_1, R_2, R_L = metric_c.ROUGE_score(a_list, true_list )
    bleu = metric_c.BLEU_1_score(a_list, true_list)
    
    for j in range(len(q_list)):
        f.write(
f'''Prompt: {q_list[j]}\nGround truth: {true_list[j]}
[{bleu[j]*100:5.2f}, {R_1[j]*100:5.2f}, {R_2[j]*100:5.2f}, {R_L[j]*100:5.2f}, {bert[j]*100:5.2f}] Response: {a_list[j]}
''' +"="*80+"\n")
        
    f.write(f"Enc:{Enc}, Policy:{Policy}\n")
    f.write(f"BLEU_1: {sum(bleu)/len(bleu)*100:5.2f}\n")
    f.write(f"ROUGE-1: {sum(R_1)/len(R_1)*100:5.2f}\n")
    f.write(f"ROUGE-2: {sum(R_2)/len(R_2)*100:5.2f}\n")
    f.write(f"ROUGE-L: {sum(R_L)/len(R_L)*100:5.2f}\n")
    f.write(f"BERT: {sum(bert)/len(bert)*100:5.2f}\n")
    if maching is not None:
        f.write(f"Exact match1: {sum(maching)/len(maching)}\n")
    