import numpy as np
import torch
import torch.utils.data as torch_data

class ACEDataset(torch_data.Dataset):
    def __init__(self, dataset):
        self.sents, self.labels = dataset

    def __getitem__(self, index):
        return [self.sents[index], self.labels[index]]

    def __len__(self):
        return len(self.sents)


def pad_seq(seqs_in):
    '''
        Function: sort in descreasing order by seq_length, padding to max_sent_length, to tensor
        Input:
            seqs_in: list of list [seq1, seq2]
        Output:
            seqs_in_batch: Tensor (batch_size, max_sent_len)
    '''

    # sort in decreasing order
    seqs_in.sort(key=lambda s: -1 * len(s))

    batch_size = len(seqs_in)
    lens = np.array([len(s) for s in seqs_in], dtype=np.int64)
    max_sent_len = max(lens)

    #make batch
    seqs_in_batch = np.zeros((batch_size, max_sent_len), dtype=np.int64)

    for i in range(batch_size):
        l = lens[i]
        seqs_in_batch[i, :l] = seqs_in[i]
    seqs_in_batch = torch.from_numpy(seqs_in_batch)
    lens = torch.from_numpy(lens)
    return seqs_in_batch, lens

def pad_trig(batch):
    '''
        Input:
            batch: list of [sent, target]
        Output:
            sentences_in_batch, targets_in_batch, lens
            sentences_in_batch: LongTensor, (batch_size, max_sent_len)
            targets_in_batch: LongTensor, (batch_size, max_sent_len)
            lens: LongTensor, (batch_size, )
            mask: ByteTensor, (batch_size, max_sent_len)
    '''
    sentences_in = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    sentences_in_batch, lens = pad_seq(sentences_in)
    targets_in_batch, _ = pad_seq(targets)

    (batch_size, max_sent_len) = sentences_in_batch.size()
    mask = np.zeros((batch_size, max_sent_len), dtype=np.int64)
    for i in range(batch_size):
        l = lens[i]
        mask[i, :l] = [1 for _ in range(l)]
    mask = torch.from_numpy(mask).byte()
    return sentences_in_batch, targets_in_batch, lens, mask
