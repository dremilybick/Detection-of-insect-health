import pandas as pd

df = pd.read_csv('strict_length_fixed.csv')
df = df[['file_name', 'insect_name', 'type', 'index', 'thres_signal', 'strict_length', 'max', 'snippets']]

testlist = df.sample(frac=0.1)
trainlist = df[~df['index'].isin(testlist['index'])]

samplelist = testlist.sample(n=20)

trainlist.to_csv('siamese_embeddings/train.txt', index=False)
testlist.to_csv('siamese_embeddings/test.txt', index=False)
samplelist.to_csv('siamese_embeddings/sample.txt', index=False)




