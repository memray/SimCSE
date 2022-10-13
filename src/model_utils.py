import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import transformers
from transformers.file_utils import (
    ModelOutput,
)

import torch
from torch import nn
from src import contriever

logger = logging.getLogger(__name__)


def load_retriever(model_id, pooling, hf_config=None):
    if not hf_config:
        hf_config = load_hf(transformers.AutoConfig, model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=model_id, config=hf_config,
                                                   add_pooling_layer=False)
    retriever = contriever.Contriever(tokenizer, model, hf_config, pooling)

    if 'bert' in model_id:
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token = "[CLS]"
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = "[SEP]"
    elif 't5' in model_id:
        raise NotImplementedError()
    retriever.cls_token_id = tokenizer.cls_token_id

    return retriever, tokenizer


def load_hf(object_class, model_name):
    try:
        obj = object_class.from_pretrained(model_name, local_files_only=True)
    except:
        obj = object_class.from_pretrained(model_name, local_files_only=False)
    return obj


def gather_norm(input, input_mask=None):
    if input_mask is not None:
        _norm = torch.linalg.norm((input * input_mask.unsqueeze(-1)), dim=1)
        _norm = torch.masked_select(_norm, input_mask.bool().reshape(-1))
    else:
        _norm = torch.linalg.norm(input, dim=1, ord=2)
    return _norm.mean()


@dataclass
class ContrastiveLearningOutput(ModelOutput):
    """
    Base class for outputs of sentence contrative learning models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    specific_losses: Optional[dict] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GaussianDropout(nn.Module):
    '''
    Gaussian Dropout from: Fast dropout training
    https://nlp.stanford.edu/pubs/sidaw13fast.pdf
    '''
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x

class VariationalDropout(nn.Module):
    '''
    Variational Dropout from: Variational Dropout and the Local Reparameterization Trick
    https://arxiv.org/pdf/1506.02557.pdf
    '''
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = torch.randn(x.size())
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x