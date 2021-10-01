import os
os.environ['CUDA_VISIBLE_DEVICES']="0"

from model import STN4DST
from pytorch_transformers import BertTokenizer, AdamW, WarmupLinearSchedule, BertConfig
from utils.ckpt_utils import download_ckpt
from evaluation import model_evaluation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import argparse
import random
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss

def masked_cross_entropy_for_bio(logits, target, pad_idx=19):
    total_loss = None
    for i in range(logits.size(0)):
        logits_i = logits[i]
        target_i = target[i]
        gather_position = torch.tensor([j for j in range(target_i.size(0)) if target_i[j] != pad_idx]).to(logits.device)
        gather_position1 = gather_position[:, None].expand(-1, logits_i.size(-1))
        gather_logits_i = torch.gather(logits_i, 0, gather_position1)
        gather_target_i = torch.gather(target_i, 0, gather_position)
        logits_i_flat = gather_logits_i.view(-1, gather_logits_i.size(-1))
        log_probs_flat = torch.log(logits_i_flat)
        gather_target_i_flat = gather_target_i.view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=gather_target_i_flat)
        losses = losses_flat.view(*gather_target_i.size())
        mask = gather_target_i.ne(pad_idx)
        losses = losses * mask.float()
        loss = losses.sum() / (mask.sum().float())
        total_loss = loss if total_loss is None else total_loss + loss
    return total_loss

def main(args):
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)
    
    if args.dataset == 'sim-R':
        from utils.simR_data_utils import prepare_dataset, MultiWozDataset, make_turn_label, postprocessing, state_equal, SLOT, BIO
    if args.dataset == 'sim-M':
        from utils.simM_data_utils import prepare_dataset, MultiWozDataset, make_turn_label, postprocessing, state_equal, SLOT, BIO
    if args.dataset == 'DSTC2':
        from utils.DSTC2_data_utils import prepare_dataset, MultiWozDataset, make_turn_label, postprocessing, state_equal, SLOT, BIO
    if args.dataset == 'WOZ2.0':
        from utils.WOZ_data_utils import prepare_dataset, MultiWozDataset, make_turn_label, postprocessing, state_equal, SLOT, BIO
    if args.dataset == 'MultiWOZ2.2':
        from utils.MultiWOZ_data_utils import prepare_dataset, MultiWozDataset, make_turn_label, postprocessing, state_equal, make_slot_ontology, BIO
        schema = json.load(open(args.schema_data_path))
        SLOT, bio_ontology = make_slot_ontology(schema, args.train_data_path)

    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    slot_meta = SLOT
    bio_meta = BIO
    print(len(slot_meta))
    print(len(bio_meta))
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    train_data_raw = prepare_dataset(data_path=args.train_data_path,
                                     tokenizer=tokenizer,
                                     slot_meta=slot_meta,
                                     bio_meta=bio_meta,
                                     n_history=args.n_history,
                                     max_seq_length=args.max_seq_length,
                                     data_type='train')

    # for data in train_data_raw[:50]:
    #     print([[j,d] for j,d in enumerate(data.input_)])
    #     print([[j,d] for j,d in enumerate(data.input_bio)])
    #     print(data.last_dialog_state)
    #     print(data.turn_dialog_state)
    #     print(data.op_labels)
    #     print('\n\n\n')
    # exit()

    train_data = MultiWozDataset(train_data_raw,
                                 tokenizer,
                                 bio_meta,
                                 args.max_seq_length,
                                 args.word_dropout,
                                 args.value_dropout)
    print("# train examples %d" % len(train_data_raw))

    dev_data_raw = prepare_dataset(data_path=args.dev_data_path,
                                   tokenizer=tokenizer,
                                   slot_meta=slot_meta,
                                   bio_meta=bio_meta,
                                   n_history=args.n_history,
                                   max_seq_length=args.max_seq_length,
                                   data_type='dev')
    print("# dev examples %d" % len(dev_data_raw))

    test_data_raw = prepare_dataset(data_path=args.test_data_path,
                                    tokenizer=tokenizer,
                                    slot_meta=slot_meta,
                                    bio_meta=bio_meta,
                                    n_history=args.n_history,
                                    max_seq_length=args.max_seq_length,
                                    data_type='test')
    print("# test examples %d" % len(test_data_raw))

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model_config.hidden_dropout_prob = args.hidden_dropout_prob
    model = STN4DST(model_config, len(slot_meta), len(bio_meta))

    if not os.path.exists(args.bert_ckpt_path):
       args.bert_ckpt_path = download_ckpt(args.bert_ckpt_path, 'bert-base-uncased')
    ckpt = torch.load(args.bert_ckpt_path, map_location='cpu')
    ckpt1 = {k.replace('bert.', '').replace('gamma','weight').replace('beta','bias'): v for k, v in ckpt.items() if 'cls.' not in k}
    model.bert.load_state_dict(ckpt1)

    # re-initialize added special tokens ([SLOT], [NULL], [APPD])
    model.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
    model.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    model.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
    model.to(device)

    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = WarmupLinearSchedule(optimizer, int(num_train_steps * args.warmup),
                                         t_total=num_train_steps)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)

    loss_fnc = nn.CrossEntropyLoss()
    best_score = {'epoch': 0, 'joint_acc': 0, 'op_acc': 0, 'final_slot_f1': 0}
    total_step = 0
    for epoch in range(args.n_epochs):
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = [b.to(device) for b in batch]
            input_ids, input_mask, segment_ids, op_ids, bio_ids = batch

            state_scores, bio_scores = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            loss_s = loss_fnc(state_scores.contiguous().view(-1, state_scores.size(-1)), op_ids.view(-1))
            loss_bio = masked_cross_entropy_for_bio(bio_scores.contiguous(), bio_ids.contiguous(), len(BIO))
            
            loss = loss_s + loss_bio

            batch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            total_step += 1

            if step % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, bio_loss : %.3f" \
                        % (epoch, args.n_epochs, step, len(train_dataloader), np.mean(batch_loss), loss_s.item(), loss_bio.item()))
                batch_loss = []
            # break

        if epoch % args.eval_epoch == 0:
            print('total_step: ',total_step)
            eval_res = model_evaluation(make_turn_label, postprocessing, state_equal, model, dev_data_raw, tokenizer, slot_meta, bio_meta, epoch)
            if eval_res['joint_acc'] > best_score['joint_acc']:
                best_score = eval_res
                model_to_save = model.module if hasattr(model, 'module') else model
                save_path = os.path.join(args.save_dir,  'model_best_seed[%s]_history[%s].bin'% (args.random_seed, str(args.n_history)))
                torch.save(model_to_save.state_dict(), save_path)
            print("Best Score : ", best_score)
            print("\n")
        
        if best_score['epoch']+args.patience < epoch:
                print("out of patience...")
                break
        
    print("Test using best model...")
    best_epoch = best_score['epoch']
    ckpt_path = os.path.join(args.save_dir, 'model_best_seed[%s]_history[%s].bin'% (args.random_seed, str(args.n_history)))
    model = STN4DST(model_config, len(slot_meta), len(bio_meta))
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)
    model_evaluation(make_turn_label, postprocessing, state_equal, model, test_data_raw, tokenizer, slot_meta, bio_meta, best_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MultiWOZ2.2', type=str)
    parser.add_argument("--vocab_path", default='bert-base-uncased/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='bert-base-uncased/bert_config_base_uncased.json', type=str)
    parser.add_argument("--bert_ckpt_path", default='./bert-base-uncased/pytorch_model.bin', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--lr", default=4e-5, type=float)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)
    parser.add_argument("--patience", default=15, type=int)

    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--value_dropout", default=0.0, type=float)

    parser.add_argument("--n_history", default=0, type=int)
    parser.add_argument("--max_seq_length", default=220, type=int)

    args = parser.parse_args()
    if args.dataset == 'sim-R':
        data_root = 'data/M2M/sim-R'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.save_dir = 'outputs/sim-R_outputs'
    elif args.dataset == 'sim-M':
        data_root = 'data/M2M/sim-M'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.save_dir = 'outputs/sim-M_outputs'
    elif args.dataset == 'DSTC2':
        data_root = 'data/DSTC2'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.save_dir = 'outputs/DSTC2_outputs'
    elif args.dataset == 'WOZ2.0':
        data_root = 'data/WOZ2.0'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.save_dir = 'outputs/WOZ_outputs'
    elif args.dataset == 'MultiWOZ2.2':
        data_root = 'data/MultiWOZ_2.2'
        args.train_data_path = os.path.join(data_root, 'train')
        args.dev_data_path = os.path.join(data_root, 'dev')
        args.test_data_path = os.path.join(data_root, 'test')
        args.schema_data_path = os.path.join(data_root, 'schema.json')
        args.save_dir = 'outputs/MultiWOZ_outputs'
    else:
        print('select dataset in sim-R, sim-M, DSTC2, WOZ2.0 and MultiWOZ2.2')
        exit()

    print('pytorch version: ', torch.__version__)
    print('dataset: ', args.dataset)
    print(args)
    main(args)
