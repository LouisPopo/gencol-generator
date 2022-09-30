import numpy as np
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
import warnings
warnings.filterwarnings("ignore")
from model_dgl import MDEVSPGraphDataset, Model, train_model, eval_model, ModelAttention, BCELoss_class_weighted, ModelAttentionSimple, ModelSoftmax
import wandb
# ## Apprentissage
list_graphs = np.array(['4a_10_'+str(i) for i in range(500)])
dossier = '../MDEVSP/MdevspGencolTest/gcn_4_10/'
nom_model = 'model_27'
rang = 20
cc = True

np.random.shuffle(list_graphs)
train, validation, test = np.split(list_graphs, [int(.7*len(list_graphs)),int(.85*len(list_graphs))])

batch_size = 5
label = 'sol_int'
dataloader_train = GraphDataLoader(
    MDEVSPGraphDataset(train,label,dossier, rang, cc), batch_size=batch_size, drop_last=False, shuffle=True)
dataloader_eval = GraphDataLoader(
    MDEVSPGraphDataset(validation,label,dossier, rang, cc), batch_size=batch_size, drop_last=False, shuffle=True)
dataset_test = MDEVSPGraphDataset(test,label,dossier, rang, cc)
dataloader_test = GraphDataLoader(
    dataset_test, batch_size=batch_size, drop_last=False, shuffle=True)

g = dataset_test[0]
node_features = g.ndata['features'].float()
edge_features = g.edata['features'].float()
edge_label = g.edata['label']

config = {
    'epochs': 200,
    'batch_size': batch_size,
    'lr': 1e-4,
    'log_every': 10,  # Number of batches between each wandb logs
    'patience' : 5,  # Patience for early stopping mechanism
}
config['train_loader'] = dataloader_train
config['val_loader'] = dataloader_eval
config['d_n'] = node_features.shape[1]
config['d_e'] = edge_features.shape[1]
config['d_h'] = 128
config['L'] = 25
#config['K'] = 10
#config['M'] = 8
config['dropout'] = False
model = Model(config['d_n'], config['d_e'], config['d_h'], config['L'], config['dropout'])
config['optimizer'] = torch.optim.Adam(model.parameters(), lr=config['lr'])
config['wandb'] = True
config['dataset'] = (list_graphs[0],len(list_graphs))
config['model'] = model
config['type_model'] = 'model'
config['loss_fn'] = BCELoss_class_weighted([0.1, 1.0]) # nn.BCELoss() # 
config['label'] = label
config['nom_model'] = nom_model 
config['type_dataset'] = '_'.join(list_graphs[0].split('_')[:2])
config['len_dataset'] = len(list_graphs)
config['rang'] = rang
config['cc'] = cc

with wandb.init( config=config, project='maitrise_recherche', group='model_dgl'):
    model = train_model(model, config)

with open(nom_model+"/out.txt","w") as f:
    f.write(str(config))
    f.write('\n\n')
    f.write(str(eval_model(model, dataloader_test, config['loss_fn'])))

with open(nom_model+"/list_test.py",'w') as f :
    f.write('list_test = [')
    for pb in test:
        f.write('\''+pb+'\''+', ')
    f.write(']\n')
    
with open(nom_model+"/list_train.py",'w') as f :
    f.write('list_train = [')
    for pb in train:
        f.write('\''+pb+'\''+', ')
    f.write(']\n')
    
with open(nom_model+"/list_eval.py",'w') as f :
    f.write('list_eval = [')
    for pb in validation:
        f.write('\''+pb+'\''+', ')
    f.write(']\n')

torch.save(model.state_dict(), nom_model+'/model.pt')
