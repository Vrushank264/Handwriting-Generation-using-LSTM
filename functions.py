import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fun

class OneHotEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, array, mask):
        shape1 = array.size() + (self.vocab_size, )
        encoded_array = torch.zeros(shape1).float().cuda()
        encoded_array.scatter_(-1, array.unsqueeze(-1), 1)
        one_hot_array = encoded_array * mask.unsqueeze(-1)
        return one_hot_array


def mixture_of_bivariate_normal_nll(data,log_pi, myu, log_sigma, rho, eps=1e-6):

    '''
    Inputs are,
    data: x and y points
    log_pi: mixture weights
    myu: means of bivariate normal distribution (contains 2 means: myu1 & myu2)
    log_sigma: variance of bivariate normal distribution (contains 2 variances: log_sigma1 & log_sigma2 )
    rho: Covariance 
    Output: 
    Negative log likelihood
    '''
    
    x, y = data.unsqueeze(-2).unbind(-1)
    myu1, myu2 = myu.unbind(-1)
    log_sigma1, log_sigma2 = log_sigma.unbind(-1)
    sigma1 = log_sigma1.exp() + eps
    sigma2 = log_sigma2.exp() + eps
    
    Z = torch.pow((x - myu1) / sigma1, 2) + torch.pow((y - myu2) / sigma2, 2)
    Z -= 2 * rho * ((x - myu1) * (y - myu2)) / (sigma1 * sigma2)
    
    logN = -Z / (2 * (1 - rho ** 2) + eps)
    logN -= np.log(2 * np.pi) + log_sigma1 + log_sigma2
    logN -= 0.5 * torch.log(1 - rho ** 2 + eps)
    nll = -torch.logsumexp(log_pi + logN, dim = -1) #logsumexp to accurately compute log probability of mixture normal distribution -> log(sigma (x_ij))
    return nll


def mixture_of_bivariate_normal_sample(log_pi, myu, log_sigma, rho,  bias, eps=1e-6,):
    
    batch_size = log_pi.shape[0]
    ndims = log_pi.dim()
    
    if ndims > 2:
        
        log_pi, myu, log_sigma, rho = [x.reshape(-1, *x.shape[2:]) for x in [log_pi, myu, log_sigma, rho]]
        
    pi = log_pi.exp() * (1 + bias)
    
    #Sample mixture index using mixture weights probabilities pi
    mixture_index = pi.multinomial(1).squeeze(1) #Multinomial returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input
    
    myu, log_sigma, rho = [x[torch.arange(mixture_index.shape[0]), mixture_index] for x in [myu, log_sigma, rho]]
    
    #Calculate Biased Variances
    sigma = (log_sigma - bias).exp()
    
    #Sample from the bivariate Normal Distribution
    myu1, myu2 = myu.unbind(-1)
    sigma1, sigma2 = sigma.unbind(-1)
    z1, z2 = torch.randn_like(myu1), torch.randn_like(myu2)
    
    x = myu1 + sigma1 * z1
    y = myu2 + sigma2 * (z2 * ((1 - rho**2) ** 0.5) + z1 * rho)
    
    sample = torch.stack([x,y], 1)
    
    if ndims > 2:
        sample = sample.view(batch_size, -1, 2)
        
    return sample

class GaussianWindow(nn.Module):
    
    def __init__(self, hidden_size, num_mixtures, attention_constant = .05):
        
        super(GaussianWindow, self).__init__()
        
        self.linear = nn.Linear(hidden_size, 3 * num_mixtures)
        self.num_mixtures = num_mixtures
        self.attention_constant = attention_constant
        self.init_parameters()
        
    def init_parameters(self):
        for param in self.parameters():
            if isinstance(param, nn.Linear):
                nn.init.trunc_normal_(param.weight, std=0.075)    
    
    def forward(self, h_t, Kappa_tm1, char_seq, char_mask):
        
        B, T, _ = char_seq.shape
        device = char_seq.device
        
        alpha, beta, kappa = torch.exp(self.linear(h_t))[:, None].chunk(3, dim = -1) #(B,1,K)
        kappa = kappa * self.attention_constant + Kappa_tm1.unsqueeze(1)
        
        u = torch.arange(T, dtype = torch.float32).to(device)
        u = u[None, :, None].repeat(B, 1, 1)
        phi = alpha * torch.exp(-beta * torch.pow(kappa - u, 2))
        phi = phi.sum(-1) * char_mask
        
        w = (phi.unsqueeze(-1) * char_seq).sum(1)
        
        attention_variables = {
            'alpha': alpha.squeeze(1),
            'beta': beta.squeeze(1),
            'kappa': kappa.squeeze(1),
            'phi': phi
            }
        
        return w, attention_variables
    
    
