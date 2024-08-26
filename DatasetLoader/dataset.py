import torch
from torch.utils.data import Dataset, DataLoader
import json,re
from tqdm import tqdm
import h5py
import numpy as np
from transformers import AutoTokenizer
import math
from threading import Thread, Event,Lock
import multiprocessing
from queue import Queue
import time,os,gc
import re
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


class LazySlice:
    def __init__(self, parent, slice_obj):
        self.parent = parent
        self.slice_obj = slice_obj
        self.indices = range(*slice_obj.indices(len(self.parent.dataset)))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return LazySlice(self.parent, slice(self.indices[idx].start, self.indices[idx].stop, self.indices[idx].step))
        return self.parent[self.indices[idx]]

    def __iter__(self):
        for idx in self.indices:
            yield self.parent[idx]

    def __len__(self):
        return len(self.indices)
    
def write_segment(f, s2c:Queue, qlock:Lock, flock:Lock, event:Event):
    buffer_size = 16
    while True:
        qlock.acquire()
        if not s2c.empty():
            segment_data = []
            for _ in range(min(buffer_size, s2c.qsize())):
                segment_data.append(s2c.get())

            qlock.release()
            segment_data = torch.stack(segment_data)
            while not flock.acquire(timeout=0.1):
                time.sleep(0.1)

            try:
                f['segments'].resize((f['segments'].shape[0] + segment_data.shape[0]), axis=0)
                f['segments'][-segment_data.shape[0]:, :] = segment_data
            finally:
                flock.release()

            del segment_data

        else:
            qlock.release()
            if event.is_set():
                break

def generate_segments(text, window_size, step)-> torch.Tensor:

    tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    tokens: torch.Tensor
    segment_list=[]

    for i in range(0, max(len(tokens)-window_size,1), step):
        segment_data = tokens[max(0, min(i, len(tokens)-window_size)):i+window_size]
        # print(segment_data.shape)
        if len(segment_data) < window_size:
            # print('\n\nsome thing wrong please check generate seg', len(tokens))
            padding = torch.zeros(window_size - len(segment_data), dtype=torch.long)
            segment_data = torch.cat((segment_data, padding))
        segment_list.append(segment_data)
    segment_list=torch.stack(segment_list)
    return  segment_list

def process_data(args):
    text, url, window_size, step, output_lock, output_file = args
    segment_data=generate_segments(text, window_size, step)
    with output_lock:
        try:
            f=h5py.File(output_file, 'a')
            if 'segments' not in f:
                f.create_dataset('segments', data=segment_data, maxshape=(None, window_size), dtype='i')
            else:
                f['segments'].resize((f['segments'].shape[0] + segment_data.shape[0]), axis=0)
                f['segments'][-segment_data.shape[0]:, :] = segment_data
        finally:
            f.close()

def segmentation_para(shared_dict, file_lock, shared_int, data, window_size, step, output_file='app/data/segmented_data.h5'):
    gc.collect()
    docu_ids = shared_dict
    first=True
    # output_lock = Lock()
    manager=multiprocessing.Manager()
    output_lock=manager.Lock()

    num_processes = 3
    # num_processes=1
    pool = multiprocessing.Pool(processes=num_processes)
    # results=[]
    bar = tqdm(data)
    for text, url in bar:
        with file_lock:
            if url in docu_ids:
                shared_int.value += 1
                continue
            else:
                docu_ids.update({url: False})
        if first:
            first=False
            try:
                print(f"{os.getpid()} Create Dataset")
                f=h5py.File(output_file, 'w')
                segment_data=generate_segments(text, window_size, step)
                f.create_dataset('segments', data=segment_data, maxshape=(None, window_size), dtype='i')
            finally:
                f.close()
            continue
        pool.apply(process_data, args=((text, url, window_size, step, output_lock, output_file),))
        bar.set_description_str(f"Process ID:{os.getpid()} Skip: {shared_int.value} Dict length:{len(docu_ids)}")

    # 等待所有任务完成
    # for result in results:
    #     result.get()
    #     result.wait()

    pool.close()
    pool.join()

    gc.collect()


def Write_segment_Buffer(output_file, s2c:Queue, qlock:Lock, flock:Lock, event:Event):
    while True:
        qlock.acquire()
        if not s2c.empty():
            segment_data = s2c.get()
            qlock.release()
            flock.acquire()
            try:
                f=h5py.File(output_file, 'a')
                if 'segments' not in f:
                    f.create_dataset('segments', data=segment_data, maxshape=(None, segment_data.shape[1]), dtype='i')
                else:
                    f['segments'].resize((f['segments'].shape[0] + segment_data.shape[0]), axis=0)
                    f['segments'][-segment_data.shape[0]:, :] = segment_data
            finally:
                f.close()
            flock.release()
        else:
            qlock.release()
            if event.is_set() and s2c.empty():
                break
            time.sleep(0.001)


def flush_buffer_to_file(f, buffer, output_lock):
    output_lock.acquire()
    try:
        for segment_data in buffer:
            f['segments'].resize((f['segments'].shape[0] + segment_data.shape[0]), axis=0)
            f['segments'][-segment_data.shape[0]:, :] = segment_data
    finally:
        output_lock.release()


class NQADataset(Dataset):
    def __init__(self, data_path='data/cleandata.jsonl',num_samples=None, use_long=True, use_short=False, use_doc = False, file= None):
        '''
        '''
        self.data_path = data_path
        self.num_samples = num_samples
        self.use_long = use_long
        self.use_short = use_short
        self.use_doc = use_doc
        self.file = file
        self.data = self.load_data()
        
        if use_doc:
            if self.data[0].get("document", None) is None and file is None:
                raise RuntimeError("You need to input file to provide document because the training data only provide file name.")
                
    def load_data(self):
        data = []
        skip = 0
        zh = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == self.num_samples:
                    break
                line = json.loads(line)
                if not self.use_doc and line.get("document", False):
                    del line["document"]
                if self.use_long and line["long_answer"]:
                    if len(line["long_answer"])>1000 or len(line["long_answer"])<3:
                        skip+=1
                        continue
                    elif re.search(u'[\u4e00-\u9fff]', line["long_answer"]):
                        zh+=1
                        continue
                    data.append(line)
                elif self.use_short and line["short_answers"]:
                    if len(line["short_answers"][0])==0:
                        skip+=1
                        continue
                    elif re.search(u'[\u4e00-\u9fff]', line["short_answers"]):
                        zh+=1
                        continue
                    data.append(line)
                else:
                    skip+=1
        print(f"Loaded data total: {len(data)}, Skip: {skip}, Zh: {zh}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out = [self.data[idx]['question']]
        if self.use_long:
            out += [self.data[idx]['long_answer']]
        if self.use_short:
            out += [self.data[idx]['short_answers']]
        if self.use_doc:
            if self.data[0].get("document", None) is not None:
                out += [self.data[idx]['document']]
            else:
                out += [self.file[self.data[idx]['file']], self.data[idx]['file']]

        return out#str(sample['question_text']),long_answer#text_without_tags
class PretrainEnc(NQADataset):
    def __getitem__(self, idx):

        out = [self.data[idx]['question']]
        if self.use_long:
            out += [self.data[idx]['long_answer']]
        if self.use_short:
            out += [self.data[idx]['short_answers']]
        if self.use_doc:
            out += [self.data[idx]['document']]
            out += [self.data[idx]['score']]

        return out#str(sample['question_text']),long_answer#text_without_tags
    
def text_normal(text):
    if isinstance(text, list):
        return [*map(text_normal, text)]
    if isinstance(text, tuple):
        return tuple(map(text_normal, text))

    if not isinstance(text, str):
        return text
    text = text.strip()
    text=re.sub("(<[^<>]{1,20}>)", '', text)
    text = re.sub(" +", " ", text)
    replace_word = {
    "s \' ": "s\' ",
    "`` ": "\"",
    " \'\'": "\"",
    }
    for k,v in replace_word.items():
        text = text.replace(k, v)
    right = ["\'s", "\'m", "\'d", "\'ll", "\'re", "n\'t", ":", ";", ",", "\.", "\)","\%","\!","\?", "-", "--", "/"]
    left = ["\(", "-", "--", "/"]
    right = "( )("+"|".join(right)+")"
    left = "("+"|".join(left)+")( )"
    text = re.sub(right, r"\2",text)
    text = re.sub(left, r"\1",text)
    text = re.sub("\.+", ".", text)

    #<Th_colspan="2">
    return text.strip()
def segmentation(shared_dict,file_lock,shared_int,segment, window_size, step, output_file='app/data/segmented_data.h5'):
    '''load data, segmented to 288 token id, save to h5py'''

    # Token indices sequence length is longer than the specified maximum sequence length for this model (14441966 > 512).
    # 將文本分割成token, use tokenizer!!
    first=True
    max_queue_size = 512
    num_thread = 8
    seg_count=0
    docu_ids=shared_dict

    output_lock = Lock()
    qlock = Lock()
    s2c = Queue()
    terminal_signal = Event()
    # 初始化h5py文件
        # 創建一個dataset用於保存分割後的片段
    bar = tqdm(segment)
    pool=[Thread(target = Write_segment_Buffer, args = (output_file, s2c, output_lock, qlock, terminal_signal)) for _ in range(num_thread)]
    [t.start() for t in pool]

    for text, url  in bar:
        with file_lock:
            if url in docu_ids:
                shared_int.value+=1

                continue
            else:
                docu_ids.update({url:False})
                # print(len(docu_ids))

        # print(tokens)#Tensor [3214,  2005, 25439, 87,..., 2759]

        # # 計算窗格的數量
        # num_windows = int(math.ceil((len(tokens)-window_size+0) / (step)) + 1)
        # seg_count+=num_windows
        # # bar.set_description_str(f'{seg_count:7.1e}/{seg_count*len(bar)/(bar.last_print_n+1):7.1e}, skip:{skip}')
        # # print(f"Total tokens: {len(tokens)}")
        # # print(f"Num windows: {num_windows}")

        # # 分割文本並保存到dataset
        # segment_list=[]
        # for i in range(num_windows):
        #     start_idx = i * (step)
        #     end_idx = start_idx + window_size
        #     end_idx = min(end_idx, len(tokens))
        #     segment_data = tokens[start_idx:end_idx]
        #     if len(segment_data) < window_size :
        #         assert i == num_windows - 1
        #         eos_padding = torch.zeros(window_size - len(segment_data), dtype=torch.long)
        #         segment_data = torch.cat((segment_data, eos_padding))

        if first:
            segment_data=generate_segments(text, window_size, step)
            try:
                f=h5py.File(output_file, 'w')
                if 'segments' not in f:
                    f.create_dataset('segments', data=segment_data, maxshape=(None, segment_data.shape[1]), dtype='i')
            finally:
                f.close()
            first=False
            continue

        while s2c.qsize() >= max_queue_size:
            time.sleep(0.001)
        s2c.put(generate_segments(text, window_size, step))
        bar.set_description_str(f"Process ID:{os.getpid()} Skip: {shared_int.value} Dict length:{len(docu_ids)}")
    terminal_signal.set()
    [t.join() for t in pool]

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, idx):
    #     '''input id, output segment
    #     dynamic load segment
    #     please check h5py
    #     '''

    #     sample = self.data[idx]
    #     if type(sample)==list:
    #         out = [i['document_text'] for i in sample]
    #     else:
    #         out = sample['document_text']
    #     #dict_keys(['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])

    #     return out #,sample['question_text']

class QAPairDataset(Dataset):
    def __init__(self, data_path='data/cleandata.jsonl', num_samples=None):
        self.data_path = data_path
        self.num_samples = num_samples
        if num_samples is not None:
            self.data=self.data[:num_samples]
        # self.load_data()
    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == self.num_samples:
                    break  # Stop reading after reaching the desired number of samples
                # Assuming each line is a separate JSON object
                a_line = json.loads(line)
                q = a_line['question']
                a = a_line['answer']
                if type(q)==str and type(a[0])==str:
                    self.data.append([q, a])

    def __getitem__(self, idx):

        if type(idx)==int:
            q=self.data[idx]['question']
            a=self.data[idx]['short_answers']
            # q, a = self.data[idx]
            all_a=a
            a = a[np.random.randint(len(a))]
            return [q, a,all_a]
        else:
            data = []
            for i in idx:
                q=self.data[i]['question']
                a=self.data[i]['short_answers']
                # q, a = self.data[i]
                all_a=a
                a = a[np.random.randint(len(a))]
                data.append([q, a,all_a])
            return data

    def __len__(self):
        return len(self.data)

class cleanDataset(Dataset):
    def __init__(self, data_path='/home/contriever/v1.0-simplified_simplified-nq-train.jsonl',num_samples=None):
        self.data_path = data_path

        self.num_samples = num_samples

        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == self.num_samples:
                    break

                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        if 'document_text' in sample:
            document = str(sample['document_text'])
            splited_doc=document.split(" ")
        else:
            splited_doc = [sample['document_tokens'][i]["token"] for i in range(len(sample['document_tokens']))]
            document = " ".join(splited_doc)
        #dict_keys(['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])
        # dict_keys(['annotations', 'document_html', 'document_title', 'document_tokens', 'document_url', 'example_id', 'long_answer_candidates', 'question_text', 'question_tokens'])
        # print(sample['question_text'])
        a=sample['annotations'][0]['long_answer']#sample['long_answer_candidates'][random_number]
        long_answer=' '.join(splited_doc[a['start_token']:a['end_token']])
        short_annotations=sample['annotations'][0]['short_answers']
        if short_annotations==[]:
            answers = []
        else:
            if type(short_annotations)==dict:
                short_annotations=[short_annotations]

            answers=[]
            for i in range(len(short_annotations)):
                answer=' '.join(splited_doc[short_annotations[i]['start_token']:short_annotations[i]['end_token']])
                answers.append(answer)
        # print(answers)
        # print(len(sample['question_text']))
        # document = document.split("References (edit) Jump up ^")[0]
        
        document=re.sub("<H[23]> (References|Notes).*", '', document)
        answers, long_answer, document, question_text =  text_normal([answers, long_answer, document, sample['question_text']])
        return answers, long_answer, document, question_text
    

class DocumentDatasets():
    def __init__(self) -> None:
        self.file_index=6

        self.file_list = [h5py.File(f"app/data/segmented_data_{i}.h5", 'r')[u'segments'] for i in range(self.file_index)]
        self.file_len = [f.shape[0] for f in self.file_list]
        self.offset = [0]
        for i in range(0, len(self.file_len)-1):
            self.offset.append(sum(self.file_len[0:i+1]))
        self.shape = torch.Size([self.__len__(), self.file_list[0].shape[1]])

    def get_single(self, ids):

        for i in reversed(range(len(self.offset))):
            if ids >= self.offset[i]:
                return torch.tensor(self.file_list[i][ids-self.offset[i]],dtype=torch.long)

    def __getitem__(self , ids):
        if hasattr(ids, '__iter__') or type(ids)==slice:
            if type(ids)==slice:
                start, stop, step = ids.start, ids.stop, ids.step
                if step is None:
                    step=1
                if start is None:
                    start=0

                ids = range(start, stop,step)

            out=torch.empty([len(ids), self.shape[1]], dtype=torch.long)
            for i, idx in enumerate(ids):
                out[i]= self.get_single(idx)

            return out

        else:
            return self.get_single(ids)

    def __len__(self):
        #return DocumentLeng
        return sum(self.file_len)


class trivia_qadatast:
    def __init__(self, split = "train"):
        self.dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia")
        if split=="train":
            self.dataset = self.dataset["train"]
        elif split =="test":
            self.dataset = self.dataset["test"]
        elif split =="valid":
            self.dataset = self.dataset["validation"]
    def __getitem__(self, ids):
        
        if isinstance(ids, slice):
            return LazySlice(self, ids)
        #     return lambda x: self.__getitem__(range(ids.start, ids.stop, ids.step)[x])
        #['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer']
        #question=q, short_answers=ans, long_answer=la, document = d
        answer = [self.dataset[ids]["answer"]["value"] ]+ self.dataset[ids]["answer"]["aliases"]
        return text_normal([answer, None, "\n".join(self.dataset[ids]["entity_pages"]["wiki_context"]), self.dataset[ids]['question']])
    def __len__(self,):
        return len(self.dataset)

def cleandata(dataset, outpath):

    num_data=0
    total_a_lenth=[]
    file = open(outpath, 'w')
    for i in tqdm(range(len(dataset)), ncols=0):
        ans, la, d, q=dataset[i]
        if ans:
            for a in ans:
                if len(a.split())>10:
                    ans.remove(a)
                    continue
                total_a_lenth.append(len(a.split()))
        if ans or la:
            if la and len(la)>3000:
                continue
            json.dump(dict(question=q, short_answers=ans, long_answer=la, document = d), file)
            file.write('\n')
            num_data+=1
    file.close()

    print(sum(total_a_lenth)/len(total_a_lenth))
    print(max(total_a_lenth))
    print('total:',num_data)#98708

def cleandata_trivia(outpath):
    dataset=trivia_qadatast()

    num_data=0
    total_a_lenth=[]
    file = open(outpath, 'w')
    for i in tqdm(range(len(dataset)), ncols=0):
        ans, la, d, q=dataset[i]
        if ans:
            for a in ans:
                if len(a.split())>10:
                    ans.remove(a)
                    continue
                total_a_lenth.append(len(a.split()))
        if ans or la:
            if len(la)>3000:
                continue
            json.dump(dict(question=q, short_answers=ans, long_answer=la, document = d), file)
            file.write('\n')
            num_data+=1
    file.close()

    print(sum(total_a_lenth)/len(total_a_lenth))
    print(max(total_a_lenth))
    print('total:',num_data)#98708


if __name__=="__main__":
    d = trivia_qadatast()[10:]
    for i  in range(3):
        print(d[i][0][0])
