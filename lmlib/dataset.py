from typing import Tuple
import torch
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab.vocab import Vocab


def get_batch(source: torch.Tensor, i: int, seqlen: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(seqlen, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target
    

class DatasetWrapper:
    def __init__(self, root: str):
        self.root = root
        self.tokenizer, self.vocab = self._build_vocab(root)

    def get_vocab_size(self):
        return len(self.vocab)

    def _build_vocab(self, root:str):
        raise NotImplementedError

    def _build_vocab_helper(self, data):
        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(map(tokenizer, data), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        return tokenizer, vocab

    def load_and_process_data(self, batch_size: int, eval_batch_size: int, device: torch.device):
        return self._load_and_process_data_helper(self.root, batch_size, eval_batch_size, device)

    def _load_and_process_data_helper(self, root:str, batch_size: int, eval_batch_size: int, device: torch.device):
        raise NotImplementedError

    def data_process(
        self,
        raw_text_iter: dataset.IterableDataset
    ) -> torch.Tensor:

        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def data_process_str(
        self,
        raw_text: str
    ) -> torch.Tensor:

        """Converts raw text into a flat Tensor."""
        # TODO: simplify this
        data = [torch.tensor(self.vocab(self.tokenizer(raw_text)), dtype=torch.long)]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def tokens2string(self, data):
        return " ".join(self.vocab.lookup_tokens(data))


    def batchify(self, data: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Divides the data into batch_size separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Args:
            data: Tensor, shape [N]
            batch_size: int, batch size

        Returns:
            Tensor of shape [N // batch_size, batch_size]
        """
        seq_len = data.size(0) // batch_size
        data = data[:seq_len * batch_size]
        data = data.view(batch_size, seq_len).t().contiguous()
        return data.to(device)


class WikiText2Wrapper(DatasetWrapper):

    def __init__(self, root: str = "/home/kate/Code/datasets"):
        super().__init__(root)


    def _build_vocab(self, root: str):
        train_data = WikiText2(root=root, split='train')
        return super()._build_vocab_helper(train_data)

    def _load_and_process_data_helper(self, root:str, batch_size: int, eval_batch_size: int, device: torch.device):
        
        if eval_batch_size <= 0:
            eval_batch_size = batch_size

        train_iter, val_iter, test_iter = WikiText2(root=root)
        train_data = self.batchify(self.data_process(train_iter), batch_size, device)
        val_data = self.batchify(self.data_process(val_iter), eval_batch_size, device)
        test_data = self.batchify(self.data_process(test_iter), eval_batch_size, device)
        return train_data, val_data, test_data


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    wiki = WikiText2Wrapper("/home/kate/Code")
    train, val, test = wiki.load_and_process_data(batch_size=20, eval_batch_size=10, device=device)
    print(train.shape, val.shape, test.shape)