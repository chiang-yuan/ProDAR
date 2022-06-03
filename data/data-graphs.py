#!/usr/bin/env python
# coding: utf-8

# # **Data-graph**
# This script processes PDB-GO pairs filtered by *data-sifts*, implements normal mode analysis (NMA), persistence homology (PH), and persistence image (PI), and exports proteins as JSON graphs.
# 
# ## **0. Prerequisites**
# ### Required packages

# In[2]:


import os
import json
import time
import math
import copy
import requests
import numpy as np
import scipy as sp
import gudhi as gd
import gudhi.representations
import networkx as nx
from prody import *
from pylab import *
from argparse import Namespace
from matplotlib import pyplot as plt
from matplotlib import colors as mplcolor
from IPython.display import clear_output
from tqdm.auto import tqdm

import py3Dmol


# ### Configurations

# In[5]:


config = Namespace(
    # display
    demo=False,
    demo_num=10,
    dpi=100,
    
    img_width=200,
    img_height=150,
    
    sleep=0.5,
    clear=False,
    
    # pdb-go pairs
    sifts_file='./sifts/pdbmfgos-thres-50.json',
    
    # prody
    prody_verbose='none',
    prody_pdbdir='./pdbs',
    atom_select='calpha', # pdb atom selection (backbone, calpha)
    
    # persistence homology
    simplex='alpha', # simplicial complex method (alpha, rips, witness, cover, tangential)
    pi_range=[0, 50, 0, 50*math.sqrt(2)/2], # TODO: resize
    pi_size=[25, 25], # persistence image size
    pi_dir='./pis',
    
    # normal mode analysis
    nma_cutoff=10,
    nma_gamma=1.0,
    n_modes=20,
    n_cpus=4,
    corr_thres=0.5,
    
    nma_dir = './nma-anm',
    
    # networkx
    node_size=15,
    edge_width=0.5,
    shadow_shift=0.01,
    font_size=8,
    graph_dir=f'./graphs',
    
    # figure
    fig_dir = '../../paper/dar/figures',
    fig_dpi = 300,
)

config.graph_dir=f'./graphs-{config.nma_cutoff}A'

prody.confProDy(verbosity=config.prody_verbose)
prody.pathPDBFolder(folder=config.prody_pdbdir, divided=False)


# ### Utilities

# In[6]:



table = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 
         'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
         'LEU':10, 'LYS':11, 'MET':12, 'PHE':13, 'PRO':14,
         'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19, 
         'ASX': 3, 'GLX': 6, 'CSO': 4, 'HIP': 8, 'HSD': 8,
         'HSE': 8, 'HSP': 8, 'MSE':12, 'SEC': 4, 'SEP':15,
         'TPO':16, 'PTR':18, 'XLE':10, 'XAA':20}

def draw_pdpi(pers, pi):
    with plt.style.context('default'):
        (fig_width, fig_height) = plt.rcParams['figure.figsize']

        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        WIDTH = 1

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


        fig, axes = plt.subplots(figsize=(fig_width*0.9, fig_height*0.45),
                                 ncols=2, nrows=1,
                                 dpi=config.dpi)

        i = 0
        uxyd = np.array(list(map(lambda x: [x[1][0], x[1][1], x[0]], pers)))
        axes[i].plot(uxyd[uxyd[:,2]==1,0], uxyd[uxyd[:,2]==1,1], '.', label=r'$H_1$')
        axes[i].plot(uxyd[uxyd[:,2]==2,0], uxyd[uxyd[:,2]==2,1], '.', label=r'$H_2$')
        xlo, xhi = axes[i].get_xlim()
        ylo, yhi = axes[i].get_ylim()
        axes[i].plot([0, max(xhi, yhi)], [0, max(xhi, yhi)], '-k', lw=WIDTH, zorder=0)
        axes[i].set(title='Persistence Diagram',
                    xlabel='Birth', ylabel='Death', 
                    xlim=(0, max(xhi, yhi)), ylim=(0, max(xhi, yhi)), aspect='equal')
        axes[i].legend(frameon=False, loc='lower right')

        i = 1
        img = axes[i].imshow(np.reshape(pi, config.pi_size), 
                       cmap='nipy_spectral', norm=mplcolor.Normalize(),
                       aspect='equal', origin='lower', interpolation='lanczos')
        fig.colorbar(img, ax=axes[i])
        axes[i].set(title='Persistence Image',
                    xticks=np.linspace(0, config.pi_size[1], 6),
                    yticks=np.linspace(0, config.pi_size[0], 6),
                    xlim=(0, config.pi_size[1]-1),
                    ylim=(0, config.pi_size[0]-1))
        # axes[i].axis('off')
        fig.tight_layout(pad=0)
    return fig

def draw_adjcor(adjmtrx, cormtrx, diffmtrx, nres):
    with plt.style.context('default'):
        """
        args:
            - adjmtrx: adjacency matrix
            - cormtrx: correlation matrix
            - nres: number of residues
        """
        (fig_width, fig_height) = plt.rcParams['figure.figsize']

        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        WIDTH = 1

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


        fig, axes = plt.subplots(figsize=(fig_width*1.35, fig_height*0.45),
                                 ncols=3, nrows=1, 
                                 dpi=config.dpi)

        i = 0
        img = axes[i].imshow(adjmtrx, 
                             cmap='binary',
                             aspect='equal', origin='lower', interpolation='none')
        axes[i].set(title='Contact',
                    xlabel='Residue', ylabel='Residue',
                    xticks=np.arange(0, nres, int(nres/5)),
                    yticks=np.arange(0, nres, int(nres/5)))
        
        i = 1
        img = axes[i].imshow(cormtrx, 
                             cmap='jet', vmin=-1, vmax=1, #'RdBu'
                             aspect='equal', origin='lower', interpolation='none')
        fig.colorbar(img, ax=axes[i])
        axes[i].set(title='Correlation',
                    xlabel='Residue', ylabel='Residue',
                    xticks=np.arange(0, nres, int(nres/5)),
                    yticks=np.arange(0, nres, int(nres/5)))
        
        i = 2
        img = axes[i].imshow(diffmtrx, 
                             cmap='bwr_r', vmin=-1, vmax=1, 
                             aspect='equal', origin='lower', interpolation='none')
        axes[i].set(title='Adjacency',
                    xlabel='Residue', ylabel='Residue',
                    xticks=np.arange(0, nres, int(nres/5)),
                    yticks=np.arange(0, nres, int(nres/5)))
        
        fig.tight_layout(pad=0)

    return fig

def draw_graph(graph):
    """ position """
    pos = nx.layout.spectral_layout(graph)
    pos = nx.spring_layout(graph, pos=pos, iterations=max(int(len(graph.nodes)/10),10))

    """ shadow """
    import copy
    pos_shadow = copy.deepcopy(pos)
    for idx in pos_shadow:
        pos_shadow[idx][0] += config.shadow_shift
        pos_shadow[idx][1] -= config.shadow_shift
        
    labels = nx.get_node_attributes(graph, 'resname') 
    
    with plt.style.context('default'):
        (fig_width, fig_height) = plt.rcParams['figure.figsize']

        fig, ax = plt.subplots(figsize=(fig_height*0.5, fig_height*0.5),
                                 dpi=config.dpi, frameon=False)
        
        ax.axis('off')

        nx.draw_networkx_nodes(graph, pos_shadow, node_size=config.node_size, node_color='k', alpha=0.5, 
                               ax=ax)
        nx.draw_networkx_nodes(graph, pos, 
                               node_size=config.node_size, 
                               node_color=list(map(lambda x: table[x], labels.values())), cmap=plt.cm.gist_rainbow, 
                               vmin=min(table.values()), vmax=max(table.values()),
                            #    node_color=range(len(pos)), cmap=plt.cm.gist_rainbow, 
                               ax=ax)
        nx.draw_networkx_edges(graph, pos, width=config.edge_width, 
                               ax=ax)
        # nx.draw_networkx_labels(graph, pos, labels=labels, font_size=config.font_size, 
        #                         ax=ax)
    
    return fig

def internet_on():
    try:
        requests.get('https://ftp.wwpdb.org')
        return True
    except Exception as err:
        print('Error: {}'.format(err))
        return False


# ## **1. Make necessary directories**

# In[7]:


if not os.path.exists(config.prody_pdbdir):
    os.makedirs(config.prody_pdbdir, exist_ok=True)
    
if not os.path.exists(config.graph_dir):
    os.makedirs(config.graph_dir, exist_ok=True)
    
if not os.path.exists(config.pi_dir):
    os.makedirs(config.pi_dir, exist_ok=True)
    
if not os.path.exists(config.nma_dir):
    os.makedirs(config.nma_dir, exist_ok=True)


# ## **2. Process and export**

# In[ ]:


with open(config.sifts_file, "r") as in_file:
    pdbgos = json.load(in_file)

print("protein entries: {:6d}\ngo entries: {:6d}".format(
    len(pdbgos.keys()), 
    max(list(map(lambda x: [max(x)] if x else [], list(pdbgos.values()))))[0]+1)
)

pdbfreqs = {}
pdbcontcorredges = {}

for i, polymer in enumerate(tqdm(pdbgos.keys())):
    
    if config.demo and i >= config.demo_num:
        break
        
    pdb_id, chain_id = polymer.split('_')
    
    # Retrieve PDB 
    
    atoms = None

    while atoms is None:
        if internet_on():
            #with suppress(Exception):
            try:
                atoms = parsePDB(pdb_id, subset=config.atom_select, chain=chain_id)
            except Exception as err:
                print('Error: {}'.format(err))
                break
                pass
            else:
                break
        else:
            print ("Resetting network adapters...")
            dwnnw = 'ifconfig wlo1 down'
            upnw = 'ifconfig wlo1 up'
            os.system(dwnnw)
            os.system(upnw)
            dwnnw = 'ifconfig eno2 down'
            upnw = 'ifconfig eno2 up'
            os.system(dwnnw)
            os.system(upnw)
            time.sleep(config.sleep)
            continue

    if atoms is None:
        print(f'Skipping polymer {polymer} due to error when parsing PDB...')
        continue

    if config.demo:
        view1 = prody.showProtein(atoms, width=config.img_width, height=config.img_height)
        view1.show()
        
    # ====================
    # Persistence Homology
    # ====================
    
    """ simplicial complex """
    if config.simplex == 'alpha':
        scx = gd.AlphaComplex(points=atoms.getCoords().tolist()).create_simplex_tree()
    elif config.simplex == 'rips':
        distmtrx = sp.spatial.distance_matrix(atoms.getCoords().tolist(), atoms.getCoords().tolist(), p=2)
        scx = gd.RipsComplex(distance_matrix=distmtrx).create_simplex_tree()
    
    """ persistence """
    pers = scx.persistence()
    
    """ persistence image """
    PI = gd.representations.PersistenceImage(
        bandwidth=1, weight=lambda x: max(0, x[1]**2),
        im_range=config.pi_range, 
        resolution=config.pi_size
    )
    if scx.persistence_intervals_in_dimension(1).size != 0 and scx.persistence_intervals_in_dimension(2).size != 0:
        pi = PI.fit_transform([np.concatenate((scx.persistence_intervals_in_dimension(1), 
                                               scx.persistence_intervals_in_dimension(2)), axis=0)])
    elif scx.persistence_intervals_in_dimension(1).size != 0 and scx.persistence_intervals_in_dimension(2).size == 0:
        pi = PI.fit_transform([scx.persistence_intervals_in_dimension(1)])
    elif scx.persistence_intervals_in_dimension(1).size == 0 and scx.persistence_intervals_in_dimension(2).size != 0:
        pi = PI.fit_transform([scx.persistence_intervals_in_dimension(2)])
    else:
        continue
    
    if not config.demo:
        np.save('{:s}/{:s}.npy'.format(config.pi_dir, polymer), pi)
    
    if config.demo:
        draw_pdpi(pers, pi)
        # plt.savefig(f'{config.fig_dir}/ph-{polymer}.png', dpi=config.fig_dpi)
        plt.show()
    
    # ====================
    # Normal Mode Analysis
    # ====================
    
    anm = ANM(name=pdb_id)
    
    anm.buildHessian(atoms, cutoff=config.nma_cutoff, gamma=config.nma_gamma, n_cpu=config.n_cpus, norm=True)
    
    """ Kirchhoff matrix """
    K = anm.getKirchhoff()

    D = np.diag(np.diag(K) + 1.)

    """ Contact map """
    cont = -(K - D)
    
    """ Mode calculation """
    try:
        anm.calcModes(n_modes=config.n_modes, zeros=False, turbo=True)
    except Exception as err:
        print(err)
        continue
    
    freqs = []
    for mode in anm:
        freqs.append(math.sqrt(mode.getEigval()))
    
    pdbfreqs[polymer] = freqs
    
    """ Correlation map """
    corr = calcCrossCorr(anm)

    corr_abs = np.abs(corr)
    corr_abs[corr_abs < config.corr_thres] = 0
    diff = cont - corr_abs
    diff[diff < 0] = -1
    diff[diff >= 0] = 0
    
    pdbcontcorredges[polymer] = [int(np.sum(cont)), int(-np.sum(diff))]
    
    
    """ Adjacency matrix """
    
    comb = cont + diff # 1: contact edge / -1: correlation edge
    Adj = np.abs(comb)
    
    if not config.demo:
        writeNMD(
            '{:s}/{:s}.nma'.format(config.nma_dir, polymer), 
            modes=anm, 
            atoms=atoms
        )
    
    if config.demo:
        draw_adjcor(cont, corr, comb, nres=atoms.numResidues())
        # plt.savefig(f'{config.fig_dir}/map-{polymer}.png', dpi=config.fig_dpi)
        plt.show()
            
    # =====
    # Graph
    # =====
            
    g = nx.from_numpy_array(Adj)
    g.graph['pdbid'] = pdb_id
    g.graph['chainid'] = chain_id
    
    """ node attributes """
    
    attrs = {}
    for i, resname in enumerate(atoms.getResnames()):
        attrs[i] = {}
        attrs[i]["resname"] = resname
    
    nx.set_node_attributes(g, attrs)
    
    """ edge attributes """
    
    for edge in g.edges:
        node_i, node_j = edge        
        if comb[node_i][node_j] == 1:
            g.edges[edge]["weight"] = 1   # contact map
        elif comb[node_i][node_j] == -1:
            g.edges[edge]["weight"] = -1  # correlation map

    ''' map from serial id to residue id '''
    mapping = dict(zip(g, atoms.getResnums().tolist()))
    g = nx.relabel.relabel_nodes(g, mapping)
    
    json_graph = nx.readwrite.json_graph.node_link_data(g)
    
    if not config.demo:
        with open(config.graph_dir+"/{:s}.json".format(polymer), "w") as out_file:
            json.dump(json_graph, out_file, 
                      skipkeys=False, 
                      ensure_ascii=True, 
                      indent=None, separators=(', ', ': '),
                      sort_keys=True)
    
    if config.demo:
        draw_graph(g)
        # plt.savefig(f'{config.fig_dir}/graph-{polymer}.png', dpi=config.fig_dpi)
        plt.show()
        time.sleep(config.sleep)
        if config.clear:
            clear_output(wait=True)

    del atoms
    del scx, pers, PI
    del anm, K, D, cont, corr, diff, comb, Adj
    del g, json_graph


# ## **3. Summary**

# In[ ]:


if not config.demo:
    with open("pdbfreqs.json", "w") as out_file:
        json.dump(pdbfreqs, out_file, 
                  skipkeys=False, 
                  ensure_ascii=True, 
                  indent=None, separators=(', ', ': '),
                  sort_keys=True)
        
    with open("pdbedges.json", "w") as out_file:
        json.dump(pdbcontcorredges, out_file, 
                  skipkeys=False, 
                  ensure_ascii=True, 
                  indent=None, separators=(', ', ': '),
                  sort_keys=True)

