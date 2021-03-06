import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

Tab = "\t"

def tensor2var(eg_tensor):
    var = autograd.Variable(eg_tensor, requires_grad=False)
    return var

def init_embedding(dim1, dim2):
    init_embeddings = np.random.uniform(-0.01, 0.01, (dim1, dim2))
    return np.matrix(init_embeddings)


def idsent2words(sent, vocab):
    id2word = dict([(word_index, word) for word, word_index in vocab.items()])
    words = [id2word.get(word_index) for word_index in sent]
    return words

def get_trigger(sent_tags):
    triggers = [(word_idx, tag) for word_idx, tag in enumerate(sent_tags) if tag != 0]
    return triggers

def set_loss_optim(parameters, loss_flag, opti_flag, lr):
    if loss_flag == 'nllloss':
        loss_function = nn.NLLLoss()
    else:
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    if opti_flag == "adadelta":
        optimizer = optim.Adadelta(parameters, lr=lr, rho=0.95)
        #optimizer = optim.Adadelta(parameters, lr=lr)
    elif opti_flag == "sgd":
        optimizer = optim.SGD(parameters, lr=lr)
    elif opti_flag == "adam":
        optimizer = optim.Adam(parameters, lr=lr)
    return loss_function, optimizer

def load_models(model_path):
    model_list = torch.load(model_path)
    return model_list

def cal_prf(common, num, num_gold):
    if common == 0: return 0.0, 0.0, 0.0
    pre = common*100.0/num
    rec = common*100.0/num_gold
    f1 = 2*pre*rec/(pre+rec)
    arr = ["%.2f"%i if i != "Nan" else 0.0 for i in [pre, rec, f1]]
    return arr
