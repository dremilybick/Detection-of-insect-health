import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from siamese_embeddings.augment import smooth
import io
import os

if os.path.basename(os.getcwd())=='siamese_embeddings':
    prefix = ''
else:
    prefix = 'siamese_embeddings/'


remap = False
n_per_grp = 150
run_id = 7
interactive = False

prefix += f'runs/siamesespectral_run_{run_id}/'

df= pd.read_csv(f'{prefix}embeddings.csv')

if interactive:
    if 'smoothed' in df:
        df['smoothed'] = df['smoothed'].apply(eval)
    else:
        df['snippets'] = df['snippets'].apply(eval)
        df['smoothed'] = None
        for i, row in df.iterrows():
            analytic_signal = smooth(row['snippets'])
            envelope = np.abs(analytic_signal)
            df.at[i,'smoothed'] = list(envelope)

        df.to_csv(f'{prefix}embeddings.csv')

# Assuming df is your DataFrame
# Select only the columns containing the 64-dimensional embeddings
embedding_columns = [str(x) for x in list(range(64))]#+['strict_length','max']
embedding_data = df[embedding_columns].values

# Perform UMAP

#%%

if 'strict_length' in embedding_columns:
    prefix = prefix+'enriched'

if not os.path.exists(f'{prefix}umap_data.csv') or remap:
    umap_model = umap.UMAP(n_neighbors=50, min_dist=1, n_components=2, random_state=42)
    umap_result = umap_model.fit_transform(embedding_data)

    # Create a new DataFrame with UMAP results and metadata columns
    umap_df = pd.DataFrame(data=umap_result, columns=['UMAP_1', 'UMAP_2'])
    umap_df[['insect_name', 'type']] = df[['insect_name', 'type']]
    umap_df.loc[umap_df['type']=='Control','insect_name']='Control'

    umap_df.to_csv(f'{prefix}umap_data.csv', index=False)
else:
    umap_df = pd.read_csv(f'{prefix}umap_data.csv')



# Plot UMAP

def graphimage(data):
    plt.ioff()
    _, a = plt.subplots()
    a.plot(data)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    plt.close()
    plt.ion()
    return image

out_umap = pd.DataFrame()
for i, name in enumerate(umap_df['insect_name'].unique()):
    subset = umap_df[umap_df['insect_name']==name].sample(n=n_per_grp)
    out_umap = pd.concat([out_umap, subset])

fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='insect_name', data=out_umap, alpha=0.7)

if interactive:
    line, = ax.plot(out_umap['UMAP_1'],out_umap['UMAP_2'], ls="", marker="o", alpha=0)
    # create the annotations box
    im = OffsetImage(umap_df[['UMAP_1','UMAP_2']], zoom=0.5)
    xybox=(50., 50.)
    ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
# add callback for mouse moves
    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            if len(line.contains(event)[1]["ind"])>1:
                ind, = [line.contains(event)[1]["ind"][0]]
            else:
                ind, = line.contains(event)[1]["ind"]
            # get the figure size
            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0]*ws, xybox[1]*hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy =(umap_df['UMAP_1'][ind], umap_df['UMAP_2'][ind])
            # set the image corresponding to that point
            graph = graphimage(df['smoothed'][ind])
            im.set_data(graph)
        else:
            #if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.ion()

plt.title('UMAP Visualisation of 64-dimensional Embeddings')
plt.xlabel('UMAP_1')
plt.ylabel('UMAP_2')
plt.show()

#%%

# fix, axs = plt.subplots(1,5, figsize=(10,2))
# axs[0].plot(row['snippets'])
# axs[1].plot(np.abs(row['snippets']))
# axs[2].plot(gaussian_filter1d(np.abs(row['snippets']), sigma=3))
# axs[3].plot(row['hilbert'])
# axs[4].plot(gaussian_filter1d(row['hilbert'], sigma=3))
#
# axs[0].set_title('Raw snippet')
# axs[1].set_title('Absolute value')
# axs[2].set_title('Gaussian filtered')
# axs[3].set_title('Hilbert envelope')
# axs[4].set_title('Gaussian filtered hilbert')
#
# plt.tight_layout()

# fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharey=True, sharex=True)
# for i, name in enumerate(umap_df['insect_name'].unique()):
#     subset = umap_df[umap_df['insect_name']==name].sample(n=150)
#     ax[i].scatter(subset['UMAP_1'], subset['UMAP_2'])
#     ax[i].set_title(name)
# plt.show()