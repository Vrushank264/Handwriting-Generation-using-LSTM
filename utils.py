import torch
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.functional import Tensor

alphabet = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]
ordered_alphabets = list(map(ord, alphabet))    #ord gives unicode values of given string
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))
#print(alpha_to_num)
num_to_alpha = dict(enumerate(ordered_alphabets))
#print(num_to_alpha)


coords = np.array(([2186,3950],[2176,3957],[2175,3965],[2177,3977],[2185,3988],[2197,4001],[2204,4011],
                   [2207,4019],[2206,4019],[2195,4022],[2179,4023],[2164,4027]))
t = np.array(([1,2,1],[0.93,1.3,0],[0.90,0.92,1],[1.2,1.0,0],[1.5,1.5,0],[1.8,1.9,1]))

def align(coords):
    
    co1 = np.copy(coords)
    X, Y = co1[:, 0].reshape(-1, 1), co1[:, 1].reshape(-1, 1)
    X = np.concatenate([np.ones([X.shape[0],1]), X], axis = 1)
    offset, slope = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y).squeeze()
    theta = np.arctan(slope)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
        ])

    co1[:, :2] = np.dot(co1[:, :2], rotation_matrix) - offset
    return co1

def skew(coords, degree):
    
    co = np.copy(coords)
    theta = degree * (np.pi/180)
    ar = np.array([[np.cos(-theta),0],[np.sin(-theta),1]])
    co[:,:2] = np.dot(co[:,:2], ar)
    return co

#skew(coords, -18)


def encode_ascii(ascii_str):
    
    return np.array(list(map(lambda x: alpha_to_num[x], ascii_str)) + [0])

#encode_ascii('Vrushank Changawala')
#norm gives a magnitude or distance (frobenious norm = Euclidien distance)
#Sum over all the matrix columns and take the maximum

def normalize(offsets):
    
    offsets = np.copy(offsets)
    offsets[:,:2] /= np.median(np.linalg.norm(offsets[:, :2], axis = 1))
    return offsets

#t1 = np.arange(0, 27).reshape((9,3))
#print(t1)

def coords2offset(coords):
    
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis = 1)
    offsets = np.concatenate([np.array([[0,0,1]]), offsets], axis=0)
    return offsets

#offsets = coords2offset(t1)

def offsets2coords(offsets):
    
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis = 1)

def interpolate(coords, factor = 2):
    
    #Below line detects splits if end of stroke 
    coords = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis = 0)
    new_coords = []
    
    for stroke in coords:
        
        if len(stroke) == 0:
            continue
        
        xy_coords = stroke[:, :2]
        
        if len(stroke) > 3:
            
            fx = interp1d(np.arange(len(stroke)), stroke[:, 0], kind = 'cubic')
            fy = interp1d(np.arange(len(stroke)), stroke[:, 1], kind = 'cubic')
            
            xx = np.linspace(0, len(stroke) - 1, factor*(len(stroke)))
            yy = np.linspace(0, len(stroke) - 1, factor*(len(stroke)))
            
            x_n = fx(xx)
            y_n = fy(yy)
            xy_coords = np.hstack([x_n.reshape(-1,1), y_n.reshape(-1,1)])
        
        stroke_eos = np.zeros([len(xy_coords), 1])
        stroke_eos[-1] = 1.0
        stroke = np.concatenate([xy_coords, stroke_eos], axis=1)
        new_coords.append(stroke)
        
    coords = np.vstack(new_coords)
    return coords

@torch.no_grad()
def plot_grad_flow(named_params):

    avg_grads, max_grads, layers = [], [], []
    plt.figure(figsize = ((10,20)))
    
    for n, p in named_params:
        
        if (p.requires_grad) and ('bias' not in n):
            
            layers.append(n)
            avg_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
        
    plt.bar(np.arange(len(max_grads)), max_grads, alpha = 0.1, lw = 1, color = 'c')
    plt.bar(np.arange(len(max_grads)), avg_grads, alpha = 0.1, lw = 1, color = 'b')
    plt.hlines(0, 0, len(avg_grads) + 1, lw = 2, color = 'k')
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation = 'vertical')
    plt.xlim(left = 0, right = len(avg_grads))
    plt.ylim(bottom = -0.001, top = 0.02) #Zoom into the lower gradient regions
    plt.xlabel('Layers')
    plt.ylabel('Average Gradients')
    plt.title('Gradient Flow')
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color = 'c', lw = 4),
        Line2D([0], [0], color = 'b', lw = 4),
        Line2D([0], [0], color = 'k', lw = 4),
        ],
        ['max-gradient', 'mean-gradient','zero-gradient'])
    
    plt.savefig('/content/drive/MyDrive/Samples/grad.png')
    plt.close()

def concate_dict(main, new):
    for key in main.keys():
        main[key] += [new[key]]
        
def plot_lines(arr):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    
    for i in range(arr.shape[0]):
        ax.plot(arr[i], label = '%d' % i)
    ax.legend()
    return fig

def plot_img(arr):
    
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    img = ax.imshow(arr, origin = 'lower', aspect = 'auto', interpolation = 'nearest')
    fig.colorbar(img)
    return fig
    
def draw(offsets, ascii_seq=None, save_file=None):
    strokes = np.concatenate(
        [offsets[:, 0:1], np.cumsum(offsets[:, 1:], axis=0)],
        axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 3))

    stroke = []
    for eos, x, y in strokes:
        stroke.append((x, y))
        if eos == 1:
            xs, ys = zip(*stroke)
            ys = np.array(ys)
            ax.plot(xs, ys, 'k', c='blue')
            stroke = []

    if stroke:
        xs, ys = zip(*stroke)
        ys = np.array(ys)
        ax.plot(xs, ys, 'k', c='blue')
        stroke = []

    ax.set_xlim(-50, 600)
    ax.set_ylim(-40, 40)
    ax.axis('off')

    ax.set_aspect('equal')
    ax.tick_params(
        axis='both', left=False, right=False,
        top=False, bottom=False,
        labelleft=False, labeltop=False,
        labelright=False, labelbottom=False
    )

    if ascii_seq is not None:
        if not isinstance(ascii_seq, str):
            ascii_seq = ''.join(list(map(chr, ascii_seq)))
        plt.title(ascii_seq)

    if save_file is not None:
        plt.savefig(save_file)

    return fig
