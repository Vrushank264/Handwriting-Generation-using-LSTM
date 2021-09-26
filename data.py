import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from utils import draw

def preprocess_batch(batch):
    
    strokes, sentences = zip(*batch)
    stroke_len = [len(x) for x in strokes]
    sentences_len = [len(x) for x in sentences]
    batch_size = len(batch)
    
    stroke_arr = torch.zeros(batch_size, max(stroke_len), 3).float()
    stroke_mask = torch.zeros(batch_size, max(stroke_len)).float()
    
    sentence_arr = torch.zeros(batch_size, max(sentences_len)).long()
    sentence_mask = torch.zeros(batch_size, max(sentences_len)).float()
    
    for i, (stroke, length) in enumerate(zip(strokes, stroke_len)):
        stroke_arr[i, :length] = stroke
        stroke_mask[i, :length + 50] = 1.0
    
    for i, (sentence, length) in enumerate(zip(sentences, sentences_len)):
        sentence_arr[i, :length] = sentence
        sentence_mask[i, :length] = 1.0
        
    return sentence_arr, sentence_mask, stroke_arr, stroke_mask

class GetDataset(Dataset):
    
    def __init__(self, root, split = 'train'):
        
        super(GetDataset).__init__()
        root = Path(root)
        self.strokes = np.load(root / 'strokes.npy', allow_pickle = True, encoding = 'latin1')
        self.sentences = open(root / 'sentences.txt').read().splitlines()
        self.sentences = [list(x + ' ') for x in self.sentences]
        assert len(self.strokes) == len(self.sentences), "Dataset Incorrect!"
        
        np.random.seed(211)
        idxs = np.arange(len(self.strokes))
        np.random.shuffle(idxs)
        self.strokes = self.strokes[idxs]
        self.sentences = np.asarray(self.sentences)[idxs].tolist()
        
        c = Counter()
        for line in self.sentences:
            c.update(line)
            
        self.vocab = sorted(list(c.keys()))
        self.vocab_size = len(self.vocab)
        self.char2idx = {x: i for i, x in enumerate(self.vocab)}
        
        if split == 'train':
            self.strokes = self.strokes[:-500]
            self.sentences = self.sentences[:-500]
        else:
            self.strokes = self.strokes[-500:]
            self.sentences = self.sentences[-500:]
            
    def __len__(self):
        return self.strokes.shape[0]
    
    def sentence2idx(self, sentence):
        return np.asarray([self.char2idx[x] for x in sentence])
    
    def idx2sentence(self, sentence):
        return ''.join(self.vocab[i] for i in sentence)
    
    def __getitem__(self, idx):
        stroke = self.strokes[idx]
        stroke = torch.from_numpy(stroke).clamp(-50,50)
        #stroke = torch.from_numpy(stroke).float()
        #stroke[:, 1:] /= 20. 
        sentence = torch.from_numpy(self.sentence2idx(self.sentences[idx])).long()  
        return stroke, sentence
    

if __name__ == '__main__':
    
    
    #Test
    dataset = GetDataset('E:/NLP/Data/processed_data')
    loader = DataLoader(dataset, batch_size = 16, collate_fn = preprocess_batch)
    for i, data in tqdm(enumerate(loader)):
        data = [x.cuda() for x in data]
        (sent, sent_mask, stroke, stroke_mask) = data
        if i == 0:
            print(stroke.shape)
            print(sent.shape)
            print(stroke_mask.shape)
            print(sent_mask.shape)
            
            for x in range(16):
                print(dataset.idx2sentence(sent[x].tolist()))
                draw(stroke[x].cpu().numpy())
