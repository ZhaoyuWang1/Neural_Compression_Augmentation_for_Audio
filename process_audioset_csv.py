import pandas as pd
import os
def audioset_process_csv(input_path, target_path, fname='balanced_train_segments_processed.csv'):
  
  df = pd.read_csv(input_path, header=None, on_bad_lines='skip')
  df.iloc[:,3] =  df.apply(lambda row: '#'.join(str(elem) for elem in row[3:15] if pd.notna(elem)), axis=1)
  df.iloc[:,3] = df.iloc[:,3].str.replace('"', '')
  df = df[~df.iloc[:,0].isin(['#NAME?'])]
  selected = df.iloc[:,0:4]
  selected.to_csv(os.path.join(output_path, fname), header=False, index=False)


input_path = '/content/balanced_train_segments.csv'
output_path = '/content/save'

audioset_process_csv(input_path, output_path)
