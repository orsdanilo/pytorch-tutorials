import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from .utils import batchify


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

bptt = 35

batch_size = 20
eval_batch_size = 10
