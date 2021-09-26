import torch
import torch.nn as nn
import torch.nn.functional as fun
from functions import *
from utils import *

class HandwritingGenerator(nn.Module):
    
    def __init__(self, vocab_size, hidden_size, num_layers, num_mixtures_attn, num_mixtures_output):
        
        super(HandwritingGenerator, self).__init__()
        self.onehot = OneHotEncoder(vocab_size)
        self.lstm0 = nn.LSTMCell(input_size = 3 + vocab_size, hidden_size = hidden_size)
        self.lstm1 = nn.LSTM(input_size = 3 + vocab_size + hidden_size, hidden_size = hidden_size, batch_first = True)
        self.lstm2 = nn.LSTM(3 + vocab_size + hidden_size, hidden_size, batch_first = True)
        self.window = GaussianWindow(hidden_size = hidden_size, num_mixtures = num_mixtures_attn)
        self.fc = nn.Linear(hidden_size * 3, num_mixtures_output * 6 + 1)
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_mixtures_output = num_mixtures_output
        self.init_parameters()
        
    def initialize_hidden(self, batch_size):
        
        hid0 = torch.zeros(batch_size, self.hidden_size * 2).float().cuda()
        hid0 = hid0.chunk(2, dim = -1)
        hid1, hid2 = None, None
        w0 = torch.zeros(batch_size, self.vocab_size).float().cuda()
        k0 = torch.zeros(batch_size, 1).float().cuda()
        
        return hid0, hid1, hid2, w0, k0
    
    def init_parameters(self):
        for param in self.parameters():
            if isinstance(param, nn.Linear) or isinstance(param, nn.LSTMCell) or isinstance(param, nn.LSTM):
                nn.init.trunc_normal_(param.weight, std=0.075)
    
    def parse_outputs(self, output):
        
        K = self.num_mixtures_output
        
        myu, log_sigma, pi, rho, eos = output.split([2 * K, 2 * K, K, K, 1], -1)
        
        #applying activations to constrain the values in the correct range
        log_pi = fun.log_softmax(pi, dim = -1)
        rho = torch.tanh(rho)
        eos = torch.sigmoid(-eos)
        myu = myu.view(myu.shape[:-1] + (K, 2))
        log_sigma = log_sigma.view(log_sigma.shape[:-1] + (K, 2))
        
        return log_pi, myu, log_sigma, rho, eos
    
    def forward(self, char_seq, char_mask, strokes, strokes_mask, prev_states = None):
        
        char_seq = self.onehot(char_seq, char_mask)
        
        if prev_states is None:
            hid0, hid1, hid2, w_t, k_t = self.initialize_hidden(char_seq.size(0))
        else:
            hid0, hid1, hid2, w_t, k_t = prev_states
            
        lstm0_out, attention_out = [], []
        monitor_variables = {'phi': [], 'alpha': [], 'beta': [], 'kappa': []}
        
        for x_t in strokes.unbind(1):
            
            hid0 = self.lstm0(torch.cat([x_t, w_t], -1), hid0)
            w_t, variables_t = self.window(hid0[0], k_t, char_seq, char_mask)
            k_t = variables_t['kappa']
            
            concate_dict(main = monitor_variables, new = variables_t)
            lstm0_out.append(hid0[0])
            attention_out.append(w_t)
            
        lstm0_out = torch.stack(lstm0_out, dim = 1)
        attention_out = torch.stack(attention_out, dim = 1)
        
        lstm1_out, hid1 = self.lstm1(torch.cat([strokes, attention_out, lstm0_out], -1), hid1)
        lstm2_out, hid2 = self.lstm2(torch.cat([strokes, attention_out, lstm1_out], -1), hid2)
        final_out = self.fc(torch.cat([lstm0_out, lstm1_out, lstm2_out], -1))
        
        output_parameters = self.parse_outputs(final_out)
        monitor_variables = {x: torch.stack(y,1) for x, y in monitor_variables.items()}
        
        return output_parameters, monitor_variables, (hid0, hid1, hid2, w_t, k_t)
    
    def sample(self, char_seq, char_mask, bias, max_len = 1000):
        
        char_seq = self.onehot(char_seq, char_mask)
        last_index = (char_mask.sum(-1) - 2).long()
        
        hid0, hid1, hid2, w_t, k_t = self.initialize_hidden(char_seq.size(0))
        x_t = torch.zeros(char_seq.size(0), 3).float().cuda()
        strokes=[]
        monitor_variables = {'phi': [], 'kappa': [], 'alpha': [], 'beta': []}
        
        for i in range(max_len):
            
            hid0 = self.lstm0(torch.cat([x_t, w_t], -1), hid0)
            w_t, variables_t = self.window(hid0[0], k_t, char_seq, char_mask)
            k_t = variables_t['kappa']
            concate_dict(main = monitor_variables, new = variables_t)
            
            _, hid1 = self.lstm1(torch.cat([x_t, w_t, hid0[0]], 1).unsqueeze(1), hid1)                      #(1, Batch_size, hidden_size)
            _, hid2 = self.lstm2(torch.cat([x_t, w_t, hid1[0].squeeze(0)], 1).unsqueeze(1), hid2)            
            
            final_out = self.fc(torch.cat([hid0[0], hid1[0].squeeze(0), hid2[0].squeeze(0)],1))
            output_parameters = self.parse_outputs(final_out)
            
            x_t = torch.cat([output_parameters[-1].bernoulli(),         #bernoullis for eos
                             mixture_of_bivariate_normal_sample(*output_parameters[:-1], bias = bias)], dim = 1)
        
            #Exit Condition
            
            phi_t = variables_t['kappa']
            inspect1 = ~torch.gt(phi_t.max(1)[1], last_index)
            
            inspect2 = torch.sign(phi_t.sum(1)).byte().bool()
            
            is_incomplete = inspect1 | inspect2
            
            if is_incomplete.sum().item() == 0:
                break
            
            x_t = x_t * is_incomplete.float().unsqueeze(-1)
            strokes.append(x_t)
            
        monitor_variables = {x: torch.stack(y,1) for x,y in monitor_variables.items()}
        return torch.stack(strokes, 1), monitor_variables
    
    def losses(self, char_seq, char_mask, strokes, strokes_mask, prev_states = None):
        
        ip_strokes = strokes[:,:-1]
        ip_strokes_mask = strokes_mask[: ,:-1]
        op_strokes = strokes[:, 1:]
        
        output_parameters, monitor_variables, prev_states = self.forward(char_seq, char_mask, ip_strokes, ip_strokes_mask, prev_states = prev_states)
        
        stroke_loss = mixture_of_bivariate_normal_nll(op_strokes[:, :, 1:], *output_parameters[:-1])
        stroke_loss = (stroke_loss * ip_strokes_mask).sum(-1).mean()
        
        #bceloss = nn.BCELoss()
        eos_loss =  fun.binary_cross_entropy(output_parameters[-1].squeeze(-1), op_strokes[:, :, 0], reduction = 'none')
        eos_loss = (eos_loss * ip_strokes_mask).sum(-1).mean()
        
        teacher_forced_sample = torch.cat([output_parameters[-1].bernoulli(),
                                           mixture_of_bivariate_normal_sample(*output_parameters[:-1], bias = 5.0)], dim = -1)
        
        return stroke_loss, eos_loss, monitor_variables, prev_states, teacher_forced_sample
    
if __name__ == '__main__':
    
    vocab_size = 60
    hidden_size = 400
    
    model = HandwritingGenerator(vocab_size, hidden_size, num_layers = 3, num_mixtures_attn = 6, num_mixtures_output = 20).cuda()
    
    char_seq = torch.randint(0, vocab_size, (4, 50)).cuda()
    char_mask = torch.ones_like(char_seq).float()
    
    strokes = torch.randn(4, 300, 3).cuda()
    strokes_mask = torch.ones(4, 300).cuda()
    
    loss = model.losses(char_seq, char_mask, strokes, strokes_mask)
    #print(loss)
    
    out, var = model.sample(char_seq, char_mask)
    print("Shape: ", out[0].shape)
    print(var)
 
