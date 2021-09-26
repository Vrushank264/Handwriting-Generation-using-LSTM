import torch
import matplotlib.pyplot as plt
from data import GetDataset
from model import HandwritingGenerator
from utils import *


MODEL_PATH = '/content/drive/MyDrive/Handwriting/model.pth'
train_dataset = GetDataset('/content/drive/MyDrive/Handwriting', split = 'train')
test_dataset = GetDataset('/content/drive/MyDrive/Handwriting', split = 'test')
model = HandwritingGenerator(vocab_size = train_dataset.vocab_size, hidden_size = 400, num_layers = 3, num_mixtures_attn = 10, num_mixtures_output = 20).cuda()
model.load_state_dict(torch.load(MODEL_PATH))


while True:
    string = input("Enter input: ") + " "
    chars = torch.from_numpy(
        test_dataset.sentence2idx(string)
    ).long()[None].cuda()
    chars_mask = torch.ones_like(chars).float().cuda()

    with torch.no_grad():
        out = model.sample(chars, chars_mask, bias = 2.0)[0].cpu().numpy()

    draw(out[0], save_file='./generated.jpg')
    plt.imshow(out[0])
    print("Generated sample...\n")
