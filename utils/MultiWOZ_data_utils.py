
import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
import os
from copy import deepcopy
from collections import OrderedDict
from .fix_label import fix_general_label_error
from .fix_value import fix_value_dict
from .fix_value import fix_time

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
appendix = ['dontcare', 'none']
BIO = ['place-B', 'place-I', 'time-B', 'time-I', 'food-B', 'food-I', 'O']

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
                slot_name = tag.split('-')[0]
                tag_name = tag.split('-')[-1]
                if tag_name == 'B':
                    bio_list = [tag]
                    for i in range(list_len-1):
                        bio_list.append(slot_name + '-' + 'I')
                else:
                    bio_list = [tag] * list_len
                if word_list[-1] in ['.', '!', ',', '?']:
                    bio_list[-1] = 'O'
        else:
            word_list = [word]
            bio_list = [tag]
            word_idx.extend([diag_len])
        diag_len += 1
        input_diag.extend(word_list)
        input_bio.extend(bio_list)
    
    for i in range(len(input_diag)):
        if 'B' in input_bio[i]:
            slot_name = input_bio[i].split('-')[0]
            slot_I = slot_name + '-' + 'I'
            for j in range(i+1,len(input_diag)):
                if input_bio[j] != slot_I:
                    break
            value = ' '.join(input_diag[i:j])
            value_start[value] = i
    return input_bio, value_start, word_idx, diag_uttr

def find_position(slot_idx, last_dialog_state, value_start, slot, value_list, tokenizer):
    position = 0
    for value in value_list:
        value_t = ' '.join(tokenizer.tokenize(value))
        if value_t in value_start:
            position = value_start[value_t]
        else:
            for ns in slot_idx:
                if ns == slot:
                    continue
                v_list = last_dialog_state.get(ns)
                if v_list:
                    if value in v_list:
                        position = slot_idx[ns]
                        break
        if position != 0:
            break
    return position

def make_turn_label(bio_t, bio_h, turn_utter, history_uttr, slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, dynamic=False):

    input_bio = []
    input_bio, value_start, word_idx, diag_uttr = make_input_bio(bio_t, bio_h, turn_utter, history_uttr, tokenizer)
    diag_len = len(input_bio)
    state_idx = {}
    state_start = diag_len
    for s in slot_meta:
        state_idx[s] = state_start
        k = s.split('-')
        value_list = last_dialog_state.get(s)
        if value_list is not None:
            k.extend(['-', value_list[0]])
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
        v_list = turn_dialog_state[k]
        v = v_list[0]
        vv_list = last_dialog_state.get(k)
        if vv_list is None:
            vv_list = []
        try:
            idx = slot_meta.index(k)
            if list(set(v_list)&set(vv_list))==[]:
                if v in appd_idx:
                    op_labels[idx] = appd_idx[v]
                else:
                    op_labels[idx] = find_position(state_idx, last_dialog_state, value_start, k, v_list, tokenizer)
        except ValueError:
            continue
    for k, v_list in last_dialog_state.items():
        vv_list = turn_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv_list is None:
                op_labels[idx] = appd_idx['none']
        except ValueError:
            continue

    return op_labels, input_bio, state_idx, appd_idx, word_idx, diag_uttr

def create_candidate(bios, diag):
    idx_value = {}
    candidate = []
    for i, bio in enumerate(bios[:len(diag)]):
        candidate = []
        if 'B' in bio:
            candidate.append(diag[i])
            slot_type = bio.split('-')[0]
            for j in range(i+1, len(diag)):
                bio_j = bios[j]
                if slot_type in bio_j:
                    candidate.append(diag[j])
                if slot_type not in bio_j:
                    break
            value = ' '.join(candidate).replace(' ##', '')
            value = value.replace(' : ', ':').replace('##', '')
            idx_value[i] = value
    return idx_value

def postprocessing(slot_meta, ops, bios, last_dialog_state, diag, input_, state_idx, appd_idx):
    idx_state = {v:k for k, v in state_idx.items()}
    idx_appd = {v:k for k, v in appd_idx.items()}
    idx_value = create_candidate(bios, diag)
    new_dialog_state = {}
    for st, op in zip(slot_meta, ops):
        if op in idx_appd:
            value = idx_appd[op]
            if value != 'none':
                new_dialog_state[st] = [value]
        elif op in idx_state:
            slot = idx_state[op]
            value = last_dialog_state.get(slot)
            if value is not None:
                new_dialog_state[st] = value
        elif op in idx_value:
            value = idx_value[op]
            new_dialog_state[st] = [value]

    return new_dialog_state

def state_equal(pred_dialog_state, gold_dialog_state, slot_meta):
    equal = True
    for slot in slot_meta:
        pred_value = pred_dialog_state.get(slot)
        gold_value = gold_dialog_state.get(slot)
        if pred_value != gold_value:
            equal = False
            if pred_value is not None and gold_value is not None:
                if pred_value[0] in gold_value:
                    equal = True

    return pred_dialog_state, equal


def make_slot_ontology(schema, train_data_path):
    noncate_SLOT = []
    bio_ontology = {'place':[], 'time':[], 'food':[]}
    for domain in schema:
        domain_name = domain['service_name']
        if domain_name not in EXPERIMENT_DOMAINS:
            continue
        for slot in domain['slots']:
            slot_name = slot['name']
            if 'possible_values' not in slot:
                continue
            if not slot['is_categorical']:
                noncate_SLOT.append(slot_name)
    for _,_,files in os.walk(train_data_path):
        for f in files:
            f_name = os.path.join(train_data_path, f)
            dials = json.load(open(f_name))
            for dial_dict in dials:
                for turn in dial_dict["turns"]:
                    for service in turn['frames']:
                        for slot in service['slots']:
                            slot_name = slot['slot']
                            slot_value_list = slot['value']
                            if not isinstance(slot_value_list, list):
                                slot_value_list = [slot_value_list]
                            if slot_name in noncate_SLOT:
                                for slot_value in slot_value_list:
                                    slot_value = slot_value.lower()
                                    # if slot_value == 'junction':
                                    #     print(turn['utterance'].strip().lower())
                                    #     print(f, dial_dict['dialogue_id'], turn['turn_id'])
                                    #     print(service)
                                    #     print('\n')
                                    bio_ontology = create_bio_ontology(bio_ontology, slot_name, slot_value)
                        if 'state' not in service:
                            continue
                        for slot_name in service['state']['slot_values']:
                            slot_value_list = service['state']['slot_values'][slot_name]
                            if slot_name in noncate_SLOT:
                                for slot_value in slot_value_list:
                                    slot_value = slot_value.lower()
                                    # if slot_value == 'junction':
                                    #     print(turn['utterance'].strip().lower())
                                    #     print(f, dial_dict['dialogue_id'], turn['turn_id'])
                                    #     print(service)
                                    #     print('\n')
                                    bio_ontology = create_bio_ontology(bio_ontology, slot_name, slot_value)
    bio_ontology = {k:sorted(v, key = lambda i:len(i.split()), reverse=True) for k,v in bio_ontology.items()}
    return noncate_SLOT, bio_ontology

def create_bio_ontology(bio_ontology, slot_name, slot_value):
    if 'name' in slot_name or 'destination' in slot_name or 'departure' in slot_name:
        if slot_value not in bio_ontology['place'] and slot_value!='dontcare':
            bio_ontology['place'].append(slot_value)
    elif 'booktime' in slot_name or 'arriveby' in slot_name or 'leaveat' in slot_name:
        if slot_value not in bio_ontology['time'] and slot_value!='dontcare':
            bio_ontology['time'].append(slot_value)
    elif 'food' in slot_name:
        if slot_value not in bio_ontology['food'] and slot_value!='dontcare':
            bio_ontology['food'].append(slot_value)
    return bio_ontology

def value_span(value, diag):
    sequences = diag
    patt = '^(' + value + ')[^a-zA-Z0-9]' + '|[^a-zA-Z0-9](' + value + ')[^a-zA-Z0-9]|' + '[^a-zA-Z0-9](' + value + ')$' + '|^(' + value + ')$'
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

def make_bio(utterance, bio_ontology, turn_bio_ontology):
    uttr_tokens = utterance.split()
    base_bio = len(uttr_tokens)*['O']

    for bio_type in turn_bio_ontology:
        for value in turn_bio_ontology[bio_type]:
            span_list = value_span(value, utterance)
            if span_list == []:
                continue
            for span in span_list:
                start = span[0]
                if start > len(base_bio)-1:
                    continue
                exclusive_end = span[1] + 1
                if base_bio[start: exclusive_end].count('O') != exclusive_end-start:
                    continue
                base_bio[start] = bio_type+'-B'
                for idx in range(start+1,exclusive_end):
                    try:
                        base_bio[idx] = bio_type+'-I'
                    except Exception as e:
                        continue
    
    for bio_type in bio_ontology:
        for value in bio_ontology[bio_type]:
            span_list = value_span(value, utterance)
            if span_list == []:
                continue
            for span in span_list:
                start = span[0]
                if start > len(base_bio)-1:
                    continue
                exclusive_end = span[1] + 1
                if base_bio[start: exclusive_end].count('O') != exclusive_end-start:
                    continue
                base_bio[start] = bio_type+'-B'
                for idx in range(start+1,exclusive_end):
                    try:
                        base_bio[idx] = bio_type+'-I'
                    except Exception as e:
                        continue

    return base_bio

def create_instance(dialog_history, state_history, bio_history, n_history, tokenizer, ti, len_turns, dialogue_id,
                    turn_id, turn_dialog_state, slot_meta, bio_meta, max_seq_length):
    
    last_dialog_state = state_history[-1]

    if len(dialog_history) > 1 and n_history > 0:
        history_uttr = dialog_history[-(1 + n_history):-1]
        history_uttr = " ; ".join(history_uttr)
        bio_h1 = bio_history[-(1 + n_history):-1]
        bio_h = []
        for i in range(len(bio_h1)):
            bio_h.extend(bio_h1[i])
            if i < len(bio_h1)-1:
                bio_h.extend(['O'])
    else:
        history_uttr = ""
        bio_h = []

    if (ti + 1) == len_turns:
        is_last_turn = True
    else:
        is_last_turn = False

    turn_utter = dialog_history[-1]

    bio_t = bio_history[-1]

    op_labels, input_bio, state_idx, appd_idx, _, _ = make_turn_label(bio_t, bio_h, turn_utter, history_uttr, slot_meta, last_dialog_state, 
                                                                                turn_dialog_state, tokenizer)

    instance = TrainingInstance(dialogue_id, turn_id, bio_t, bio_h, turn_utter, history_uttr, last_dialog_state, turn_dialog_state, 
                                op_labels, input_bio, state_idx, appd_idx, max_seq_length, slot_meta, bio_meta, is_last_turn)
    
    instance.make_instance(tokenizer)
    return instance

def prepare_dataset(data_path, tokenizer, slot_meta, bio_meta, n_history, max_seq_length, data_type=''):
    data = []
    for _,_,files in os.walk(data_path):
        for f in files:
            f_name = os.path.join(data_path, f)
            dials = json.load(open(f_name))
            for dial_dict in dials:
                state_history = []
                dialog_history = []
                last_dialog_state = {}
                bio_history = []
                pre_utterance = ''
                pre_bio = []
                for ti, turn in enumerate(dial_dict["turns"]):
                    turn_id = turn["turn_id"]
                    speaker = turn['speaker']
                    utterance = turn['utterance'].strip().lower()
                    base_bio = turn['bio']
                    turn_dialog_state = {}
                    for service in turn['frames']:
                        if 'state' not in service:
                            continue
                        for slot_name in service['state']['slot_values']:
                            slot_value_list = service['state']['slot_values'][slot_name]
                            slot_name = slot_name.lower()
                            if slot_name not in slot_meta:
                                continue
                            slot_value_list = [value.lower() for value in slot_value_list]
                            if slot_name not in turn_dialog_state:
                                turn_dialog_state[slot_name] = slot_value_list

                    if speaker == 'USER':
                        if pre_utterance == '':
                            turn_uttr = utterance
                            base_bio = base_bio
                        else:
                            turn_uttr = pre_utterance + ' ; ' + utterance
                            base_bio = pre_bio + ['O'] + base_bio
                        dialog_history.append(turn_uttr)
                        bio_history.append(base_bio)
                        state_history.append(last_dialog_state)
                        len_turns = len(dial_dict['turns'])
                        dialogue_id = dial_dict["dialogue_id"]
                        instance = create_instance(dialog_history, state_history, bio_history, n_history, tokenizer, ti, len_turns, dialogue_id, 
                        turn_id, turn_dialog_state, slot_meta, bio_meta, max_seq_length)
                        data.append(instance)

                        last_dialog_state = turn_dialog_state
                    pre_utterance = utterance
                    pre_bio = base_bio

    return data

class TrainingInstance:
    def __init__(self, ID,
                 turn_id,
                 bio_t, 
                 bio_h,
                 turn_utter,
                 dialog_history,
                 last_dialog_state,
                 turn_dialog_state,
                 op_labels,
                 input_bio,
                 state_idx, 
                 appd_idx,
                 max_seq_length,
                 slot_meta,
                 bio_meta,
                 is_last_turn):
        self.id = ID
        self.turn_id = turn_id
        self.bio_t = bio_t
        self.bio_h = bio_h
        self.turn_utter = turn_utter
        self.dialog_history = dialog_history
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.turn_dialog_state = turn_dialog_state
        self.op_labels = op_labels
        self.input_bio = input_bio
        self.state_idx = state_idx
        self.appd_idx = appd_idx
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.bio_meta = bio_meta
        self.is_last_turn = is_last_turn

    def make_instance(self, tokenizer, max_seq_length=None,
                      word_dropout=0., value_dropout=0., slot_token='[SLOT]', appd_token = '[APPD]'):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        state = []
        for s in self.slot_meta:
            state.append(slot_token)
            k = s.split('-')
            v_list = self.last_dialog_state.get(s)
            if v_list is not None:
                k.extend(['-', v_list[0]])
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

        # if len(input_) > 200:
        #     print(len(input_))

        self.segment_id = segment
        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))

        self.input_mask = input_mask
        self.op_ids = self.op_labels
        self.bio_ids = [self.bio_meta.index(bio) for bio in self.input_bio]


class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, bio_meta, max_seq_length, word_dropout=0.1, value_dropout=0.2):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.bio_meta = bio_meta
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
            bio_ids[i] = bio + [len(self.bio_meta)] * (self.max_seq_length - len(bio))
        bio_ids = torch.tensor(bio_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids, op_ids, bio_ids