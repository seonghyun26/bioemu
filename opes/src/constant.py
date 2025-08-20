FONTSIZE = 28
FONTSIZE_SMALL = 20
COLORMAP="viridis"

COLORS = [
    "#C10035",
    "#3FA796",
    "#FEC220",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#5684E9",
    "#009E73",
    "#F0E442",
    "#D3D3D3", # light gray
]

cv_name = {
    'ref': 'phi',
    'phi': 'phi',
    'AE': 'deep.node-0',
    'TAE': 'deep.node-0',
    'VDE': 'deep.node-0',
    'DeepTDA': 'deep.node-0',
    'DeepLDA': 'deep.node-0',
    'DeepTICA': 'deep.node-0',
    'TBG': 'deep.node-0',
    'TBGXYZ': 'deep.node-0',
    'TBGNOLAG': 'deep.node-0',
}

bar_labels = {
    'ref': r'$\phi$',
    'phi': r'$\phi$',
    'AE': 'MLCV',
    'TAE': 'MLCV',
    'VDE': 'MLCV',
    'DeepLDA': 'MLCV',
    'DeepTDA': 'MLCV',
    'DeepTICA': 'MLCV',
    'TBG': 'MLCV',
    'TBGXYZ': 'MLCV',
    'TBGNOLAG': 'MLCV',
}