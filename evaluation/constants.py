from os.path import join

# --------- DATA ---------
RESULTS_ROOT = '/home/data_shares/scara/graphworld/results'
MODE_1_PROCESSED_DIR = join(RESULTS_ROOT, 'mode-1', 'processed')
MODE_2_3_PROCESSED_DIR = join(RESULTS_ROOT, 'mode-2-3', 'processed')


# --------- MODELS ---------
GENERATION_FEATURES = ['AttributeMask', 'CorruptedEmbeddingsReconstruction', 'CorruptedFeaturesReconstruction']
GENERATION_STRUCTURE = ['EdgeMask', 'GAE']
GENERATION_ALL = GENERATION_FEATURES + GENERATION_STRUCTURE
AUXILIARY_ALL = ['NodeClusteringWithAlignment', 'S2GRL', 'PairwiseAttrSim', 'GraphPartitioning']
CONTRAST_SAME_SCALE = ['BGRL', 'GBT', 'GCA', 'SelfGNNPPR', 'SelfGNNSplit', 'MERIT']
CONTRAST_CROSS_SCALE = ['DeepGraphInfomax', 'GraphInfoClust', 'SUBGCON']
CONTRAST_ALL = CONTRAST_SAME_SCALE + CONTRAST_CROSS_SCALE
HYBRID_ALL = ['G_Zoom', 'MEtAl', 'MVMI_FT']
SSL_MODELS = GENERATION_ALL + AUXILIARY_ALL + CONTRAST_ALL + HYBRID_ALL

# --------- BASELINES ---------
BASELINES = ['GCN', 'GAT', 'GIN']

# --------- MODELS ABBREVIATION ---------
model_abbreviations = {
    'AttributeMask': 'AM', 'CorruptedEmbeddingsReconstruction': 'CER', 'CorruptedFeaturesReconstruction': 'CFR', 'EdgeMask': 'EM', 'GAE': 'GAE',
    'NodeClusteringWithAlignment': 'NC', 'S2GRL': 'S2GRL', 'PairwiseAttrSim': 'PAS', 'GraphPartitioning': 'GP',
    'BGRL': 'BGRL', 'GBT': 'G-BT', 'GCA': 'GCA', 'SelfGNNPPR': 'S-PPR', 'SelfGNNSplit': 'S-Split', 'MERIT': 'MERIT',
    'DeepGraphInfomax': 'DGI', 'GraphInfoClust': 'GIC', 'SUBGCON': 'SUBG-CON',
    'G_Zoom': 'G-Zoom', 'MEtAl': 'MEtAl', 'MVMI_FT': 'MVMI-FT'
}

# --------- ENCODERS ---------
ENCODERS = ['GCN', 'GAT', 'GIN']

# --------- TRAINING SCHEMES ---------
TRAINING_SCHEMES = ['PF', 'URL', 'JL']

# --------- TEST METRIC ---------
TEST_METRIC = 'test_rocauc_ovr'


