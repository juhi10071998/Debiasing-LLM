import torch
import torch.nn as nn

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        
        ## this is corresponding to the soft tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        
        ## if we are initializing from the vocab, return the first n_token from the learned vocabulary, this could give 
        ## initial bias to soft tokens, maybe we should try with random initialization too.
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        ### tokens : tensor([[50256, 50256, 50256, 50256, 50256,  1544,  2499,   355,   257]])--> soft_prompt + input
        ### tokens[:, self.n_tokens:] : tensor([[1544, 2499,  355,  257]])  ---> input tokens indices
        ### self.wte(tokens[:, self.n_tokens:]).shape : torch.Size([1, 4, 768])  ---> embedding of input tokens
        ###   self.learned_embedding.repeat(input_embedding.size(0), 1, 1).shape : torch.Size([1, 5, 768])  ---> converting it to the batch size, appending it to each of the item in batch size before the input tokens
        
        ### torch.cat([learned_embedding, input_embedding], 1).shape --> torch.Size([1, 9, 768])
        
        #import pdb; pdb.set_trace()
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)