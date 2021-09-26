import torch
import numpy as np
from torch.utils.data import DataLoader
import time
from pathlib import Path
import pickle
from tqdm import tqdm
from utils import *
from data import GetDataset, preprocess_batch
from model import HandwritingGenerator
from tensorboardX import SummaryWriter


def monitor_samples():
    
    itr = iter(sample_loader)
    for i in range(8):
        data = itr.__next__()
        chars, chars_mask, strokes, strokes_mask = [x.cuda() for x in data]
        
        with torch.no_grad():
            
            stroke_loss, eos_loss, monitor_variables, _, teacher_forced_sample = model.losses(chars, chars_mask, strokes, strokes_mask)
            gen_sample = model.sample(chars, chars_mask)[0]
            
        teacher_forced_sample = teacher_forced_sample.cpu().numpy()
        gen_sample = gen_sample.cpu().numpy()
        gen_sample_b = gen_sample_b.cpu().numpy()
        
        phi = monitor_variables.pop('phi')
        fig = plot_img(phi[0].squeeze().cpu().numpy().T)
        writer.add_figure('attention/phi_%d' % i, fig, steps)
        
        for key, val in monitor_variables.items():
            fig = plot_lines(val[0].cpu().numpy().T)
            writer.add_figure('attention/%s_%d' % (key, i), fig, steps)
        
        fig = draw(gen_sample[0], save_file = open(IMG_SAVE_DIR + f'generated_{i}.png', 'wb'))
        writer.add_figure('samples/generated_%d' % i, fig, steps)
        fig1 = draw(gen_sample[0], save_file = open(IMG_SAVE_DIR + f'generated_bias6_{i}.png', 'wb'))
        writer.add_figure('samples/generated_bias6_%d' % i, fig, steps)
        
        fig = draw(teacher_forced_sample[0], save_file = open(IMG_SAVE_DIR + f'teacher_forced_sample_{i}.png', 'wb'))
        writer.add_figure('samples/teacher_forced_sample_%d' % i, fig, steps)
        
def train(epoch):
    
    global steps
    
    loss = []
    start_time = time.time()
    for i, data in tqdm(enumerate(train_loader), leave = True, position = 0):
        
        chars, chars_mask, strokes, strokes_mask = [x.cuda() for x in data]
        seq_len = strokes.shape[1]
        prev_states = None
        
        for idx in range(1, seq_len, 1000):
            stroke_loss, eos_loss, att, prev_states, _ = model.losses(
                chars, chars_mask,strokes[:, idx - 1:idx + 1000], 
                strokes_mask[:, idx - 1:idx + 1000], prev_states)
            
            prev_states = [(x[0].detach(), x[1].detach()) if type(x) is tuple else x.detach() for x in prev_states]
            
            opt.zero_grad()
            (stroke_loss + eos_loss).backward()
            for name, p in model.named_parameters():
                if 'lstm' in name:
                    p.grad.data.clamp_(-10,10)
                elif 'fc' in name:
                    p.grad.data.clamp_(-100,100)
            opt.step()
            loss.append([stroke_loss.item(), eos_loss.item()])
            
            writer.add_scalar('stroke_loss/train', loss[-1][0], steps)
            writer.add_scalar('eos_loss/train', loss[-1][1], steps)
            steps+=1
            
        if i % 10 == 0:
            
            print("Train epoch {}, IterNo {} | ms/batch {:5.2f} | loss {}".format(epoch, i, 1000*(time.time() - start_time) / len(loss), np.asarray(loss).mean(0)))
            start_time = time.time()
            loss = []
            
def test(epoch):
    
    loss = []
    start_time = time.time()
    
    for i, data in tqdm(enumerate(test_loader), leave = True, position = 0):
        chars, chars_mask, strokes, strokes_mask = [x.cuda() for x in data]
        
        with torch.no_grad():
            stroke_loss, eos_loss, _, _, _ = model.losses(chars, chars_mask, strokes, strokes_mask)
            
        loss.append([stroke_loss.item(), eos_loss.item()])
    
    stroke_loss, eos_loss = np.asarray(loss).mean(0)
    writer.add_scalar('stroke_loss/test', loss[-1][0], steps)
    writer.add_scalar('eos_loss/test', loss[-1][1], steps)
    
    print("Test epoch {} | ms/batch {:5.2f} | loss {}".format(epoch, 1000*(time.time()- start_time) / len(loss), np.asarray(loss).mean(0)))

    
if __name__ == '__main__':

    DATA_PATH = 'E:/NLP/Data/processed_data'
    IMG_SAVE_DIR = 'E:/NLP/Samples'
    LOGS_DIR = 'E:/NLP'
    MODEL_SAVE_DIR = 'E:/NLP'
    writer = SummaryWriter(str(LOGS_DIR))
    train_dataset = GetDataset(DATA_PATH, split = 'train')
    test_dataset = GetDataset(DATA_PATH, split = 'test')
    
    train_loader = DataLoader(train_dataset, batch_size = 64, collate_fn = preprocess_batch)
    test_loader = DataLoader(test_dataset, batch_size = 64, collate_fn = preprocess_batch)
    sample_loader = DataLoader(test_dataset, batch_size = 1, collate_fn = preprocess_batch)
    
    model = HandwritingGenerator(vocab_size = train_dataset.vocab_size, hidden_size = 400, num_layers = 3, num_mixtures_attn = 10, num_mixtures_output = 20).cuda()
    print(model)
    
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    itr = iter(sample_loader)
    for i in range(8):
        data = itr.__next__()
        fig = draw(
            data[2][0].numpy(),
            save_file = open(IMG_SAVE_DIR + f'original_{i}.png', 'wb')
        )
        writer.add_figure("samples/original_%d" % i, fig, 0)

    steps = 0
    for epoch in range(1, 101):
        
        torch.cuda.empty_cache()
        print('Generating Samples...')
        start = time.time()
        monitor_samples()
        print('Took %5.3f seconds to generate samples' % (time.time() - start))
        torch.cuda.empty_cache()
        grad_flow = plot_grad_flow(model.named_parameters())
        train(epoch)
        print("Testing...")
        start = time.time()
        torch.cuda.empty_cache()
        test(epoch)
        print('Took %5.3f seconds to evaluate test set' % (time.time() - start))
    
        torch.save(model.state_dict(), open(MODEL_SAVE_DIR + 'model.pth', 'wb'))
