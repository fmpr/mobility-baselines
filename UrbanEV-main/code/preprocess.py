# Adapted file from https://github.com/IntelligentSystemsLab/UrbanEV

import os
import pandas as pd
from utils import read_data
from parse import parse_args

#settings
args = parse_args()
output_dir = '../code-transformer/dataset/UrbanEV'
os.makedirs(output_dir, exist_ok=True)
args.feat = 'occ'
args.add_feat = 'None'
args.pred_type = 'region'

# process
feat, adj, extra_feat, time = read_data(args)
data = pd.DataFrame(feat)
columns = ['OT'] + [str(i) for i in range(1, data.shape[1])]
data.columns = columns
data.insert(0, 'date', time)

# saving
output_path = os.path.join(output_dir, f'{args.feat}-{args.add_feat}.csv')
data.to_csv(output_path, index=False, header=True)
print("Process completed.")
