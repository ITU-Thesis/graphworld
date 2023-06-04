from os.path import join
import seaborn as sns

# --------- DATA ---------
RESULTS_ROOT = '/home/data_shares/scara/graphworld/results'
MODE_1_PROCESSED_DIR = join(RESULTS_ROOT, 'mode-1', 'processed')
MODE_2_3_PROCESSED_DIR = join(RESULTS_ROOT, 'mode-2-3', 'processed')
MODE_2_3_MARG_PROCESSED_DIR = join(RESULTS_ROOT, 'mode-2-3-marg', 'processed')
MODE_2_3_MARG_2_PROCESSED_DIR = join(RESULTS_ROOT, 'mode-2-3-marg-2', 'processed')
PRELIMINARY_DIR = join(RESULTS_ROOT, 'preliminary', 'processed')
MODE_1_LEARNING_CURVES_DIR = join(RESULTS_ROOT, 'mode-1_learning_curves', 'processed')


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
SINGLE_ENCODER_MODELS = ['SelfGNNPPR', 'MERIT', 'G_Zoom'] # Should be filtered away when comparing encoders

# --------- MODEL CATEGORIES ---------
GENERATION_BASED_CATEGORY = 'Generation'
AUXILIARY_CATEGORY = 'Auxiliary-property'
CONTRAST_BASED = 'Contrast'
HYBRID_CATEGORY = 'Hybrid'
SSL_CATEGORIES = [GENERATION_BASED_CATEGORY, AUXILIARY_CATEGORY, CONTRAST_BASED, HYBRID_CATEGORY]

# --------- BASELINES ---------
BASELINES = ['GCN', 'GAT', 'GIN']

# --------- MODELS ABBREVIATION ---------
model_abbreviations = {
    'AttributeMask': 'AM', 'CorruptedEmbeddingsReconstruction': 'CER', 'CorruptedFeaturesReconstruction': 'CFR', 'EdgeMask': 'EM', 'GAE': 'GAE',
    'NodeClusteringWithAlignment': 'NC', 'S2GRL': 'S2GRL', 'PairwiseAttrSim': 'PAS', 'GraphPartitioning': 'GP',
    'BGRL': 'BGRL', 'GBT': 'G-BT', 'GCA': 'GCA', 'SelfGNNPPR': 'S-PPR', 'SelfGNNSplit': 'S-Split', 'MERIT': 'MERIT',
    'DeepGraphInfomax': 'DGI', 'GraphInfoClust': 'GIC', 'SUBGCON': 'SUBG-CON',
    'G_Zoom': 'G-Zoom', 'MEtAl': 'MEtAl', 'MVMI_FT': 'MVMI-FT',
    'GCN': 'GCN', 'GAT': 'GAT', 'GIN': 'GIN'
}
model_abbreviations_inv = {v: k for k, v in model_abbreviations.items()}


# --------- ENCODERS ---------
ENCODERS = ['GCN', 'GAT', 'GIN']

# --------- TRAINING SCHEMES ---------
TRAINING_SCHEMES = ['PF', 'URL', 'JL']

# --------- TEST METRIC ---------
TEST_METRIC = 'test_rocauc_ovr'


# --------- PLOTTING  ---------
ENCODER_COLORS = {'GCN': sns.color_palette("colorblind")[0], 'GAT':sns.color_palette("colorblind")[1], 'GIN':sns.color_palette("colorblind")[2]}
TRAINING_SCHEME_COLORS = {'PF':'teal', 'URL':'darkkhaki', 'JL':'mediumorchid'}
SSL_CATEGORY_COLORS = {'Generation':'darkolivegreen', 'Auxiliary':'deepskyblue', 'Contrast':'indigo', 'Hybrid':'crimson'}
FONT = 'DejaVu Sans'


