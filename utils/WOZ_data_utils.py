import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
from .fix_woz import SEMANTIC_DICT

EXPERIMENT_DOMAINS = ["restaurant"]

SLOT=['restaurant-price','restaurant-food','restaurant-area']

BIO=['restaurant-price-B','restaurant-price-I','restaurant-food-B','restaurant-food-I','restaurant-area-B','restaurant-area-I','O']

appendix = ['dontcare']

def make_input_bio(bio_t, bio_h, turn_utter, history_uttr, tokenizer):
    diag_1 = history_uttr.split()
    diag_2 = turn_utter.split()
    diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]
    diag_2 = diag_2 + ["[SEP]"]
    diag = diag_1 + diag_2
    diag_uttr = diag
    bio_base = ['O'] + bio_h + ['O'] + bio_t + ['O']
    input_diag = []
    input_bio = []
    value_start = {}
    word_idx = []
    diag_len = 0
    for word, tag in zip(diag, bio_base):
        if word != "[CLS]" and word != "[SEP]":
            word_list = tokenizer.tokenize(word)
            list_len = len(word_list)
            word_idx.extend([diag_len] * list_len)
            if tag == 'O':
                bio_list = [tag] * list_len
            else:
                slot_name = '-'.join(tag.split('-')[:2])
                tag_name = tag.split('-')[-1]
                if tag_name == 'B':
                    bio_list = [tag]
                    for i in range(list_len-1):
                        bio_list.append(slot_name + '-' + 'I')
                    # if word_list[-1] in ['.', '!', ',', '?']:
                    #     bio_list[-1] = 'O'
                else:
                    bio_list = [tag] * list_len
                    # if word_list[-1] in ['.', '!', ',', '?']:
                    #     bio_list[-1] = 'O'
        else:
            word_list = [word]
            bio_list = [tag]
            word_idx.extend([diag_len])
        diag_len += 1
        input_diag.extend(word_list)
        input_bio.extend(bio_list)
    
    for i in range(len(input_diag)):
        if 'B' in input_bio[i]:
            slot_name = '-'.join(input_bio[i].split('-')[:2])
            slot_I = slot_name + '-' + 'I'
            for j in range(i+1,len(input_diag)):
                if input_bio[j] != slot_I:
                    break
            value = ' '.join(input_diag[i:j])
            if value not in value_start:
                value_start[value] = i
            if i > value_start[value]:
                value_start[value] = i
    return input_bio, value_start, word_idx, diag_uttr

def find_position(value_start, value, tokenizer):
    value_t = ' '.join(tokenizer.tokenize(value))
    if value_t in value_start:
        position = value_start[value_t]
    else:
        position  = 0
        if value in SEMANTIC_DICT:
            for v1 in SEMANTIC_DICT[value]:
                v1_t = ' '.join(tokenizer.tokenize(v1))
                if v1_t in value_start:
                    position = value_start[v1_t]
                    break
    return position

def make_turn_label(bio_t, bio_h, turn_utter, history_uttr, slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]
    
    input_bio = []
    input_bio, value_start, word_idx, diag_uttr = make_input_bio(bio_t, bio_h, turn_utter, history_uttr, tokenizer)
    diag_len = len(input_bio)
    state_idx = {}
    state_start = diag_len
    for s in slot_meta:
        state_idx[s] = state_start
        k = s.split('-')
        value = last_dialog_state.get(s)
        if value is not None:
            k.extend(['-', value])
            t = tokenizer.tokenize(' '.join(k))
        else:
            t = tokenizer.tokenize(' '.join(k))
            t.extend(['-', '[NULL]'])
        slot_len = len(t) + 1
        state_start += slot_len

    appd_idx = {}
    appd_start = state_start
    for appd in appendix:
        appd_idx[appd] = appd_start
        appd_start = appd_start + len(tokenizer.tokenize(appd)) + 1

    op_labels = [state_idx[s] for s in slot_meta]
    keys = list(turn_dialog_state.keys())
    for k in keys:
        v = turn_dialog_state[k]
        if v == 'none':
            turn_dialog_state.pop(k)
            continue
        vv = last_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv != v:
                if v in appd_idx:
                    op_labels[idx] = appd_idx[v]
                else:
                    if dynamic==False:
                        op_labels[idx] = find_position(value_start, v, tokenizer)
        except ValueError:
            continue

    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]

    return op_labels, gold_state, input_bio, state_idx, appd_idx, word_idx, diag_uttr


# def create_candidate(bios, diag, diag_uttr, word_idx, input_):
#     idx_value = {}
#     candidate = []
#     candidate_idx = 0
#     slot = ''
#     for i, bio in enumerate(bios[:len(diag)]):
#         word_i = word_idx[i]
#         if word_i in candidate:
#             continue
#         if '-B' in bio:
#             if candidate != []:
#                 idx_value[candidate_idx] = ' '.join([diag_uttr[c] for c in candidate])
#                 slot = ''
#                 candidate = []
#                 candidate_idx = 0
#             slot = bio.replace('-B','')
#             if word_i not in candidate:
#                 candidate_idx = i
#                 candidate.append(word_i)
#         elif bio == slot+'-I':
#             if word_i not in candidate:
#                 candidate.append(word_i)
#         else:
#             if candidate != []:
#                 idx_value[candidate_idx] = ' '.join([diag_uttr[c] for c in candidate])
#             slot = ''
#             candidate = []
#             candidate_idx = 0
#     return idx_value

# def postprocessing(slot_meta, ops, bios, last_dialog_state, diag, input_, state_idx, appd_idx, word_idx, diag_uttr):
#     gid = 0
#     idx_state = {v:k for k, v in state_idx.items()}
#     idx_appd = {v:k for k, v in appd_idx.items()}
#     idx_value = create_candidate(bios, diag, diag_uttr, word_idx, input_)
#     new_dialog_state = {}
#     for st, op in zip(slot_meta, ops):
#         if op in idx_appd:
#             value = idx_appd[op]
#             if value != 'none':
#                 new_dialog_state[st] = value
#         elif op in idx_state:
#             slot = idx_state[op]
#             value = last_dialog_state.get(slot)
#             if value is not None and value != 'none':
#                 new_dialog_state[st] = value
#         elif op in idx_value:
#             value = idx_value[op]
#             new_dialog_state[st] = value

#     return new_dialog_state

def postprocessing(slot_meta, ops, bios, last_dialog_state, diag, input_, state_idx, appd_idx, word_idx, diag_uttr):
    gid = 0
    idx_state = {v:k for k, v in state_idx.items()}
    idx_appd = {v:k for k, v in appd_idx.items()}
    new_dialog_state = {}
    for st, op in zip(slot_meta, ops):
        if op in idx_appd:
            value = idx_appd[op]
            if value != 'none':
                new_dialog_state[st] = value
        elif op in idx_state:
            slot = idx_state[op]
            value = last_dialog_state.get(slot)
            if value is not None and value != 'none':
                new_dialog_state[st] = value
        elif op < len(diag):
            start = op
            end = op+1
            for i in range(op+1, len(bios)):
                end = i
                if bios[i] != st + '-I':
                    break
            g = input_[start:end]
            gen = []
            for gg in g:
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gid += 1
            gen = gen.replace(' : ', ':').replace('##', '').replace(' . ', '.')
            new_dialog_state[st] = gen
    return new_dialog_state

def state_equal(pred_dialog_state, gold_dialog_state, slot_meta):
    equal = True
    for slot in slot_meta:
        pred_value = pred_dialog_state.get(slot)
        gold_value = gold_dialog_state.get(slot)
        if pred_value != gold_value:
            equal = False
            for s in SEMANTIC_DICT:
                if pred_value in [s]+SEMANTIC_DICT[s]:
                    for s1 in [s]+SEMANTIC_DICT[s]:
                        if s1 == gold_value:
                            equal = True
                            pred_dialog_state[slot] = s
                            break
    return pred_dialog_state, equal

def process_uttr(uttr):
    result = []
    for s in uttr.split():
        s1 = s
        if s[-1] in ['.', '!', ',', '?']:
            s1 = s[:-1] + ' ' + s[-1]
        result.append(s1)
    return ' '.join(result)

def process_state(state, slot_meta):
    result = {}
    for s_dict in state:
        if s_dict['act'] == 'inform':
            slot = s_dict['slots'][0][0]
            slot = process_slot(slot)
            if slot not in slot_meta:
                continue
            value = s_dict['slots'][0][1]
            result[slot] = value
    return result

def process_slot(slot):
    if slot == 'price range':
        slot = 'price'
    slot = EXPERIMENT_DOMAINS[0] + '-' + slot
    return slot

def value_span(value, diag):
    sequences = diag
    patt = value
    pattern = re.compile(patt)
    m = pattern.finditer(sequences)
    m = [mi for mi in m]
    result=[]
    for n in m:
        line_st = sequences[:n.span()[0]]
        start = len(line_st.split())
        slot_v = [start, start + len(value.split())-1]
        result.append(slot_v)
    return result

def fix_label(uttr, value):
    span = value_span(value, uttr)
    if span == []:
        if value in SEMANTIC_DICT:
            for v in SEMANTIC_DICT[value]:
                v_span = value_span(v, uttr)
                if v_span != []:
                    value = v
                    span = v_span
                    break
    return value, span

def make_bio_and_value_position(turn_label, uttr):
    uttr_tokens = uttr.split()
    base_bio = len(uttr_tokens)*['O']
    for label in turn_label:
        slot_name = label[0]
        if slot_name == 'request':
            continue
        slot_name = process_slot(slot_name)
        if slot_name not in SLOT:
            continue
        value = label[1]
        if value == 'dontcare':
            continue
        _, value_span = fix_label(uttr, value)
        if value_span == []:
            continue
        for span in value_span:
            start = span[0]
            if start > len(base_bio)-1:
                continue
            exclusive_end = span[1] + 1
            base_bio[start] = str(slot_name)+'-B'
            for idx in range(start+1,exclusive_end):
                base_bio[idx] = str(slot_name)+'-I'
    return base_bio

def create_instance(dialog_history, state_history, bio_history, n_history, tokenizer, ti, len_turns, dialogue_id,
                    turn_domain, turn_id, turn_dialog_state, slot_meta, max_seq_length):

    last_dialog_state = state_history[-1]

    if len(dialog_history) > 1 and n_history > 0:
        history_uttr = dialog_history[-(1 + n_history):-1]
        history_uttr = " ; ".join(history_uttr)
        bio_h1 = bio_history[-(1 + n_history):-1]
        bio_h2 = [' '.join(bio1) for bio1 in bio_h1]
        bio_h = ' O '.join(bio_h2).split()
    else:
        history_uttr = ""
        bio_h = []

    if (ti + 1) == len_turns:
        is_last_turn = True
    else:
        is_last_turn = False

    turn_utter = " ; ".join(dialog_history[-1:])

    bio_t1 = bio_history[-1:]
    bio_t2 = [' '.join(bio1) for bio1 in bio_t1]
    bio_t = ' O '.join(bio_t2).split()

    op_labels, gold_state, input_bio, state_idx, appd_idx, _, _ = make_turn_label(bio_t, bio_h, turn_utter, history_uttr, slot_meta, last_dialog_state, 
                                                                                turn_dialog_state, tokenizer)

    instance = TrainingInstance(dialogue_id,turn_domain, turn_id, bio_t, bio_h, turn_utter, history_uttr, last_dialog_state, turn_dialog_state, 
                                op_labels, gold_state, input_bio, state_idx, appd_idx, max_seq_length, slot_meta, is_last_turn)
    
    instance.make_instance(tokenizer)
    return instance

def prepare_dataset(data_path, tokenizer, slot_meta, n_history, max_seq_length, data_type=''):
    dials = json.load(open(data_path))
    data = []
    for dial_dict in dials:
        state_history = []
        dialog_history = []
        last_dialog_state = {}
        bio_history = []
        for ti, turn in enumerate(dial_dict["dialogue"]):
            turn_id = ti
            turn_domain = EXPERIMENT_DOMAINS[0]
            if data_type == 'train':
                user_uttr = turn['transcript'].lower().strip()
            else:
                user_uttr = turn['asr'][0][0].lower().strip()
            user_uttr = process_uttr(user_uttr)
            
            base_bio_usr = make_bio_and_value_position(turn['turn_label'], user_uttr)
            system_uttr = turn['system_transcript'].lower().strip()
            system_uttr = process_uttr(system_uttr)
            base_bio_sys = make_bio_and_value_position(turn['turn_label'], system_uttr)
            if system_uttr == '':
                turn_uttr = user_uttr
                base_bio = base_bio_usr
            else:
                turn_uttr = system_uttr + ' ; ' + user_uttr
                base_bio = base_bio_sys + ['O'] + base_bio_usr

            dialog_history.append(turn_uttr)
            bio_history.append(base_bio)
            turn_dialog_state = process_state(turn['belief_state'], slot_meta)
            state_history.append(last_dialog_state)
            len_turns = len(dial_dict['dialogue'])
            dialogue_id = dial_dict["dialogue_idx"]
            instance = create_instance(dialog_history, state_history, bio_history, n_history, tokenizer, ti,
                                        len_turns, dialogue_id, turn_domain, turn_id, turn_dialog_state, slot_meta, max_seq_length)
            data.append(instance)

            last_dialog_state = turn_dialog_state
    return data


class TrainingInstance:
    def __init__(self, ID,
                 turn_domain,
                 turn_id,
                 bio_t, 
                 bio_h,
                 turn_utter,
                 dialog_history,
                 last_dialog_state,
                 turn_dialog_state,
                 op_labels,
                 gold_state,
                 input_bio,
                 state_idx, 
                 appd_idx,
                 max_seq_length,
                 slot_meta,
                 is_last_turn):
        self.id = ID
        self.turn_domain = turn_domain
        self.turn_id = turn_id
        self.bio_t = bio_t
        self.bio_h = bio_h
        self.turn_utter = turn_utter
        self.dialog_history = dialog_history
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.turn_dialog_state = turn_dialog_state
        self.op_labels = op_labels
        self.gold_state = gold_state
        self.input_bio = input_bio
        self.state_idx = state_idx
        self.appd_idx = appd_idx
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn

    def make_instance(self, tokenizer, max_seq_length=None,
                      word_dropout=0., value_dropout=0., slot_token='[SLOT]', appd_token = '[APPD]'):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        state = []
        for s in self.slot_meta:
            state.append(slot_token)
            k = s.split('-')
            v = self.last_dialog_state.get(s)
            if v is not None:
                k.extend(['-', v])
                t = tokenizer.tokenize(' '.join(k))
            else:
                t = tokenizer.tokenize(' '.join(k))
                t.extend(['-', '[NULL]'])
            state.extend(t)
        
        for appd in appendix:
            state.append(appd_token)
            t = tokenizer.tokenize(appd)
            state.extend(t)

        avail_length_1 = max_seq_length - len(state) - 3
        diag_1 = tokenizer.tokenize(self.dialog_history)
        diag_2 = tokenizer.tokenize(self.turn_utter)
        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncated
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]

        if len(diag_1) == 0 and len(diag_2) > avail_length_1:
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]

        drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
        diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]
        diag_2 = diag_2 + ["[SEP]"]
        segment = [0] * len(diag_1) + [1] * len(diag_2)

        diag = diag_1 + diag_2
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        # value dropout
        if value_dropout > 0.:
            for i, b in enumerate(self.input_bio):
                if 'B' in b:
                    start = i
                    end = i+1
                    for j in range(i+1, len(self.input_bio)):
                        end = j
                        if 'I' not in self.input_bio[j]:
                            break
                    random_drop = np.random.random()
                    if random_drop < value_dropout:
                        for k in range(start, end):
                            diag[k] = '[UNK]'
        
        input_ = diag + state
        segment = segment + [1]*len(state)
        self.diag = diag
        self.diag_len = len(diag)
        self.input_ = input_

        self.segment_id = segment
        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))

        self.input_mask = input_mask
        self.op_ids = self.op_labels
        self.bio_ids = [BIO.index(bio) for bio in self.input_bio]


class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length, word_dropout=0.1, value_dropout=0.2):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.word_dropout = word_dropout
        self.value_dropout = value_dropout

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.word_dropout > 0 or self.value_dropout > 0:
            self.data[idx].make_instance(self.tokenizer, word_dropout=self.word_dropout, value_dropout=self.value_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)

        bio_ids = [b.bio_ids for b in batch]
        for i, bio in enumerate(bio_ids):
            bio_ids[i] = bio + [len(BIO)] * (self.max_seq_length - len(bio))
        bio_ids = torch.tensor(bio_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids, op_ids, bio_ids
