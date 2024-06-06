import argparse
import os.path
import torch
import sys

sys.path.append("./settings")
from cholec import get_cholec_scores
from chestx import get_chestx_scores
from mass_maps import get_mass_maps_scores
from supernova import get_supernova_scores
# from politeness import get_politeness_scores
# from emotion import get_emotion_scores


all_settings_baselines = {
    'cholec': ['patch', 'quickshift', 'watershed'],
    'chestx': ['patch', 'quickshift', 'watershed'],
    'mass_maps': ['patch', 'quickshift', 'watershed'],
    'supernova': ['chunk'],
    # 'politeness': ['words'],
    # 'emotion': ['words']
}

all_settings_methods = {
    'cholec': get_cholec_scores,
    'chestx': get_chestx_scores,
    'mass_maps': get_mass_maps_scores,
    'supernova': get_supernova_scores,
    # 'politeness': get_politeness_scores,
    # 'emotion': get_emotion_scores
}


if __name__ == "__main__": 
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("setting")
    # if len(sys.argv) > 1:
    #     parser.add_argument("baselines_list")
    args = parser.parse_args()

    print('SETTING:', args.setting)
    scores_filepath = f'results/{args.setting}/all_baselines_scores'
    if not os.path.isfile(scores_filepath):
        print('Getting scores...')
        # if len(sys.argv) > 1:
        baselines_list = all_settings_baselines[args.setting]
        get_scores = all_settings_methods[args.setting]
    
        all_baselines_scores = get_scores(baselines_list)
        # dic, where name (patch) of BL and value is the patch BL number
        
        print(all_baselines_scores)
        torch.save(all_baselines_scores, f'results/{args.setting}/all_baselines_scores')
    
        
    # for baseline in all_baselines_list: 
        
    
# generate the latex row with all results for baseline done in notebook