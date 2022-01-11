import pandas as pd
import pickle
# import numpy as np
# import json

data_file = open("../../data/raw"+"/final_hdsi_faculty_updated.csv", "r")

df = pd.read_csv(data_file)

df_test = df.sample(frac=0.1, random_state=1, ignore_index=True)

# ---------------
print('export test.csv to test/raw folder')
df_test.to_csv('../../test/raw/test_data.csv', index=True, sep='\t')

print('get number of authors in the new dataframe')
number_of_authors = len(list(df_test['HDSI_author'].unique()))

print(number_of_authors)

pickle.dump(number_of_authors, open('../../test/raw/number_of_authors.pkl', 'wb'))

# ---------------

# pd.DataFrame.from_dict(data_cfg, orient="index").transpose()


