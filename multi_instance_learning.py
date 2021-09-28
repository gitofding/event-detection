import torch
import torch.nn as nn
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class MILLoss(nn.Module):
    """
    Multi-Instance Learning CrossEntropyLoss
    """

    def __init__(self, mode='min', size_average=True, reduce=True, weight=None):
        super(MILLoss, self).__init__()
        self.raw_loss_layer = nn.CrossEntropyLoss(ignore_index=-1,reduction='none', weight=weight)
        if mode not in ['max', 'min', 'mean']:
            raise NotImplementedError("Not Implement Mode in MIL: %s" % mode)
        self.mode = mode
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):

        """
        Same as raw loss layer
        :param input:
        :param target:
        :return:
        """
        # Pre Vars
        batch_size = input.size(0)
        label_data = target.cpu().detach().numpy()
        label_set = set(label_data)
        label_size = len(label_set)
        mask = input.new_zeros(batch_size)
        # print("target:{}".format(target))
        # print(max(target))
        raw_loss = self.raw_loss_layer.forward(input=input, target=target.cuda())

        raw_loss_data = raw_loss.cpu().detach().numpy()

        if self.mode == 'mean':
            if not self.reduce:
                return raw_loss
            if self.size_average:
                return torch.mean(raw_loss)
            else:
                return torch.sum(raw_loss)
        else:
            # Find Max/Min Instance in Bag
            for label in label_set:
                label_slices = np.where(label_data == label)[0]
                label_value = raw_loss_data[label_slices]
                if self.mode == 'max':
                    label_slices_index = np.argmax(label_value)
                else:
                    label_slices_index = np.argmin(label_value)
                input_index = label_slices[label_slices_index]
                mask[input_index] = 1

            mask_loss = raw_loss * mask

            # print("    label:", target)
            # print("raw  loss:", raw_loss)
            # print("     mask:", mask)
            # print("mask loss:", mask_loss)

            if not self.reduce:
                return mask_loss
            else:
                sum_loss = torch.sum(mask_loss)
                if self.size_average:
                    return sum_loss / float(label_size)
                else:
                    return sum_loss


if __name__ == "__main__":
    device = 'cpu'
    loss_layer = MILLoss(mode='min')
    # batch=5, label=3
    x = torch.Tensor([[-1.9878, -0.0174, -0.4903],
                      [-0.3585,  0.8742, -0.2200],
                      [-0.8511,  0.1589,  1.0081],
                      [-0.2360,  0.0348, -0.5164],
                      [-1.3722,  0.1582,  1.1941]])
    y = torch.LongTensor([0, 1, 1, 2, 2])

    # loss_layer.cuda(0)
    # x.cuda(0)
    # y.cuda(0)

    loss = loss_layer.forward(x, y)
    print(loss)
