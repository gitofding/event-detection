import time
from data_util import ACEDataset, pad_trig
from transformer import Trasformer
from util import load_models, set_loss_optim, cal_prf
from multi_instance_learning import MILLoss # use MIL method
import torch
import torch.utils.data as torch_data
import torch.nn.functional as F
import sys

Tab = '\t'


def trainFunc(args, ace_data, debug=False):
    # put ace digit data into pytorch DataLoader
    train_loader = torch_data.DataLoader(ACEDataset(ace_data.train), batch_size=args.batch_size, shuffle=True,
                                         collate_fn=pad_trig)
    dev_loader = torch_data.DataLoader(ACEDataset(ace_data.dev), batch_size=args.batch_size, shuffle=False,
                                       collate_fn=pad_trig)
    test_loader = torch_data.DataLoader(ACEDataset(ace_data.test), batch_size=args.batch_size, shuffle=False,
                                        collate_fn=pad_trig)

    # init models
    # decoder = TrigRNN(args)
    decoder = Trasformer(args)
    decoder = decoder.to(args.device)
    # decoder = decoder.cuda()
    # print(decoder)
    # sys.eixt(0)
    decoder.word_embeddings.weight.data.copy_(torch.from_numpy(ace_data.pretrain_embedding))
    parameters = list(decoder.parameters())
    loss_function, optimizer = set_loss_optim(parameters, args.loss_flag, args.opti_flag, args.lr)
    loss_function_mil = MILLoss(mode='min')  # 和上面from multi_instance_learning import MILLoss对应
    # loss_function_mil = MILLoss(mode='max')
    # training
    best_f1 = -1.0
    best_epoch = -1
    for epoch in range(args.epoch_num):
        training_id = 0
        loss_all = 0
        st_time = time.time()
        k = 0
        for iteration, batch in enumerate(train_loader):
            # if iteration>30:
            #     break
            sentence_in, targets, batch_sent_lens, mask = batch
            g = max(batch_sent_lens.cpu().numpy())
            if (g > k):
                k = g
            sentence_in, targets, batch_sent_lens, mask = sentence_in.to(args.device), targets.to(
                args.device), batch_sent_lens.to(args.device), mask.to(args.device)
            decoder.zero_grad()

            tag_space = decoder(sentence_in, batch_sent_lens)
            targets_masked = targets + (1 - mask).long() * -1
            if iteration % 2 == 0:
                loss = 0.02 * loss_function_mil.forward(tag_space, targets_masked.view(-1).cpu())
            else:
                loss = loss_function(tag_space, targets_masked.view(-1))

            loss_all += loss.data.item()
            loss.backward()
            optimizer.step()
            training_id += sentence_in.size(0)
            output_loss = ["%.2f" % i for i in [loss.data.item(), loss_all]]
            print('iteration', iteration, 'loss', output_loss)

        # record best result on dev
        eval_results = eval_model(dev_loader, decoder, loss_function, args, data_flag="dev")

        current_f1 = float(eval_results[1][-1])
        # print(eval_results[1][-1])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            torch.save(decoder, args.model_path)
        print('Epoch', epoch, 'result', eval_results, "Best epoch", best_epoch, 'best_f1', best_f1)
        if epoch - best_epoch == args.early_stop: break
        decoder.train()


    eval_results = eval_model(train_loader, decoder, loss_function, args, data_flag="train")
    output_model_result(eval_results, epoch, "train", args)

    # final result on test
    best_decoder = load_models(args.model_path)
    eval_flag = "test"
    eval_results = eval_model(test_loader, best_decoder, loss_function, args, vocab=ace_data.vocab,
                              tags_data=ace_data.atag_dict, data_flag=eval_flag)
    result = output_model_result(eval_results, epoch, eval_flag, args)
    return result


def output_model_result(eval_results, epoch, eval_flag, args):
    loss, prf, prf_iden = eval_results
    f1 = prf[-1]
    loss = "%.2f" % loss
    print(eval_flag, "results, epoch", epoch, Tab, loss, time.asctime())
    print("##--Classification", Tab, prf[0], Tab, prf[1], Tab, prf[2], Tab, "##-- iden:", prf_iden[0], Tab, prf_iden[1],
          Tab, prf_iden[2])
    return [prf[0], prf[1], prf[2], prf_iden[0], prf_iden[1], prf_iden[2]]


def eval_model(data_loader, decoder, loss_function, args, vocab=None, tags_data=None, data_flag=None):
    decoder.eval()
    loss_all = 0
    common = 0
    common_iden = 0
    gold = 0
    pred = 0

    for iteration, batch in enumerate(data_loader):
        sentence_in, targets, batch_sent_lens, mask = batch
        # batch_sent_lens表示一个batch中，每一句话的长度
        sentence_in, targets, batch_sent_lens, mask = sentence_in.to(args.device), targets.to(
            args.device), batch_sent_lens.to(args.device), mask.to(args.device)
        bsize = sentence_in.size(0)  # 除了最后一个batch ,bsize都是100
        slen = sentence_in.size(1)  # slen = 一个batch中最长的句子
        decoder.zero_grad()

        tag_space = decoder(sentence_in, batch_sent_lens)

        targets_masked = targets + (1 - mask).long() * -1

        loss = loss_function(tag_space, targets_masked.view(-1)).data.item()
        pred_trig_mask_4output = mask.float().unsqueeze(2).expand(bsize, slen, args.tagset_size).contiguous().view(
            bsize * slen, -1)

        # log_softmax
        _, tag_outputs = ((F.log_softmax(tag_space, dim=1)).data * pred_trig_mask_4output).max(1)

        gold_targets = targets.cpu().data.view(bsize, slen).numpy().tolist()
        pred_outputs = tag_outputs.cpu().view(bsize, -1).numpy().tolist()


        # statistic common, pred, gold for prf
        for target_doc, out_doc in zip(gold_targets, pred_outputs):

            for wid, (gitem, oitem) in enumerate(zip(target_doc, out_doc)):

                if gitem != 0: gold += 1
                if oitem != 0: pred += 1
                if gitem == oitem and (gitem != 0): common += 1  # 分类
                if (gitem != 0) and (oitem != 0): common_iden += 1  # 识别

    print("common:", common, pred, gold)
    prf = cal_prf(common, pred, gold)
    prf_iden = cal_prf(common_iden, pred, gold)
    eval_results = [loss_all, prf, prf_iden]
    return eval_results
