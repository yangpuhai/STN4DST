import os
os.environ['CUDA_VISIBLE_DEVICES']="2"
from utils.eval_utils import compute_prf, compute_acc, compute_goal
from pytorch_transformers import BertTokenizer, BertConfig

from model import STN4DST
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import os
import time
import argparse
import json
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
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
    slot_meta = SLOT
    bio_meta = BIO
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
    data = prepare_dataset(args.test_data_path,
                           tokenizer,
                           slot_meta,
                           bio_meta,
                           args.n_history,
                           args.max_seq_length,
                           'test')

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.1
    model = STN4DST(model_config, len(slot_meta), len(bio_meta))
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    print("Test using best model...")
    model_evaluation(make_turn_label, postprocessing, state_equal, model, data, tokenizer, slot_meta, bio_meta, 0)

def model_evaluation(make_turn_label, postprocessing, state_equal, model, test_data, tokenizer, slot_meta, bio_meta, epoch):
    model.eval()

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0

    results = {}
    last_dialog_state = {}
    wall_times = []
    for di, i in enumerate(test_data):

        # if di>100:
        #     break

        if i.turn_id == '0':
            last_dialog_state = {}

        i.last_dialog_state = deepcopy(last_dialog_state)

        i.make_instance(tokenizer, word_dropout=0.)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.FloatTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        
        _, _, state_idx, appd_idx, _, _ = make_turn_label(i.bio_t, i.bio_h ,i.turn_utter, i.dialog_history, slot_meta, last_dialog_state, i.turn_dialog_state,
                                          tokenizer, dynamic=True)

        start = time.perf_counter()
        with torch.no_grad():
            s, b = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        _, op_ids = s.view(-1, s.size(-1)).max(-1)
        _, bio_ids = b.view(-1, len(bio_meta)).max(-1)

        pred_ops = [a for a in op_ids.tolist()]
        
        pred_bios = [bio_meta[a] for a in bio_ids.tolist()]

        last_dialog_state = postprocessing(slot_meta, pred_ops, pred_bios, last_dialog_state,
                                                      i.diag, i.input_, state_idx, appd_idx)
        last_dialog_state, equal = state_equal(last_dialog_state, i.turn_dialog_state, slot_meta)
        
        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, str(v)]))
        i.gold_state = []
        for k, v in i.turn_dialog_state.items():
            i.gold_state.append('-'.join([k, str(v)]))
        
        if equal:
            joint_acc += 1
        
        # else:
        #     print('\n')
        #     print('----------------------------')
        #     print('i.turn_id',i.turn_id)
        #     print('i.input_',[[i, token]for i,token in enumerate(i.input_)])
        #     print('gold_bios',i.input_bio)
        #     print('pred_bios',pred_bios[:len(i.input_bio)])
        #     print('gold_op',i.op_ids)
        #     print('pred_op',pred_ops)
        #     print('state_idx',state_idx)
        #     print('appd_idx',appd_idx)
        #     print('gold_state',i.gold_state)
        #     print('pred_state',pred_state)

        
        key = str(i.id) + '_' + str(i.turn_id)
        #results[key] = [pred_state, i.gold_state]
        results[key] = [pred_ops, last_dialog_state, i.op_labels, i.turn_dialog_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    latency = np.mean(wall_times) * 1000

    compute_goal(results, slot_meta)

    print("------------------------------")
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    json.dump(results, open('preds_%d.json' % epoch, 'w'), indent=4)
    #json.dump(results, open('preds_%d.json' % epoch, 'w'))
    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score}
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MultiWOZ2.2', type=str)
    parser.add_argument("--vocab_path", default='bert-base-uncased/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='bert-base-uncased/bert_config_base_uncased.json', type=str)
    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=280, type=int)

    args = parser.parse_args()
    if args.dataset == 'sim-R':
        data_root = 'data/M2M/sim-R'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.model_ckpt_path = 'sim-R_outputs/model_best.bin'
    elif args.dataset == 'sim-M':
        data_root = 'data/M2M/sim-M'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.model_ckpt_path = 'sim-M_outputs/model_best.bin'
    elif args.dataset == 'DSTC2':
        data_root = 'data/DSTC2'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.model_ckpt_path = 'DSTC2_outputs/model_best.bin'
    elif args.dataset == 'WOZ2.0':
        data_root = 'data/WOZ2.0'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.model_ckpt_path = 'WOZ_outputs/model_best.bin'
    elif args.dataset == 'MultiWOZ2.2':
        data_root = 'data/MultiWOZ_2.2'
        args.train_data_path = os.path.join(data_root, 'train')
        args.dev_data_path = os.path.join(data_root, 'dev')
        args.test_data_path = os.path.join(data_root, 'test')
        args.schema_data_path = os.path.join(data_root, 'schema.json')
        args.model_ckpt_path = 'outputs/MultiWOZ_outputs/model_best_seed[42].bin'
    else:
        print('select dataset in sim-R, sim-M, DSTC2,  WOZ2.0 and MultiWOZ2.2')
        exit()
    main(args)
