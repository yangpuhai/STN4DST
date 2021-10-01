import argparse
import os
import json
import re
from tqdm import trange, tqdm
from multiprocessing import Pool, freeze_support, RLock

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MultiWOZ2.2', type=str)

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

    print('dataset: ', args.dataset)
    print(args)

    if args.dataset == 'MultiWOZ2.2':
        from utils.MultiWOZ_data_utils import make_slot_ontology, create_bio_ontology, make_bio
        schema = json.load(open(args.schema_data_path))
        noncate_SLOT, bio_ontology = make_slot_ontology(schema, args.train_data_path)

        def modify_file(f_name):
            dials = json.load(open(f_name[1]))
            for i in trange(len(dials), desc="Session in "+f_name[1]+" :", position=f_name[0]):
                for turn in dials[i]["turns"]:
                    utterance = turn['utterance'].strip().lower()
                    turn_bio_ontology = {'place':[], 'time':[], 'food':[]}
                    for service in turn['frames']:
                        for slot in service['slots']:
                            slot_name = slot['slot']
                            slot_value_list = slot['value']
                            if not isinstance(slot_value_list, list):
                                slot_value_list = [slot_value_list]
                            if slot_name in noncate_SLOT:
                                for slot_value in slot_value_list:
                                    slot_value = slot_value.lower()
                                    turn_bio_ontology = create_bio_ontology(turn_bio_ontology, slot_name, slot_value)
                        if 'state' not in service:
                            continue
                        for slot_name in service['state']['slot_values']:
                            slot_value_list = service['state']['slot_values'][slot_name]
                            if slot_name in noncate_SLOT:
                                for slot_value in slot_value_list:
                                    slot_value = slot_value.lower()
                                    turn_bio_ontology = create_bio_ontology(turn_bio_ontology, slot_name, slot_value)
                    turn_bio_ontology = {k:sorted(v, key = lambda i:len(i.split()), reverse=True) for k,v in turn_bio_ontology.items()}
                    bio = make_bio(utterance, bio_ontology, turn_bio_ontology)
                    turn['bio'] = bio
            os.remove(f_name[1])
            with open(f_name[1], 'w') as f:
                json.dump(dials, f, indent=4)
        
        all_files = []
        count = 0
        for data_path in [args.train_data_path, args.dev_data_path, args.test_data_path]:
            print('Processing: ', data_path)
            for _,_,files in os.walk(data_path):
                for f in files:
                    count+=1
                    files_path = os.path.join(data_path, f)
                    files_info = [count, files_path]
                    all_files.append(files_info)
                # files = [[i, os.path.join(data_path, f)] for i,f in enumerate(files)]
        
        freeze_support()
        p = Pool(len(all_files), initializer=tqdm.set_lock, initargs=(RLock(),))
        p.map(modify_file, all_files)
        print("\n" * (len(all_files) - 2))
