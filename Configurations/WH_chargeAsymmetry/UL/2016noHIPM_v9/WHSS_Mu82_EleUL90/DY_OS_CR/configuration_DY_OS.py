# Configuration file for charge asymmetry WH_SS analysis
treeName= 'Events'

tag = 'WHSS_DY_OSCR_2016noHIPM_v9_chargeAsymmetry_Mu82_Ele90' 

# used by mkShape to define output directory for root files
outputDir = 'rootFile_OS'

# file with TTree aliases
aliasesFile = 'aliases.py'

# file with list of variables
variablesFile = 'variables.py'

# file with list of cuts
cutsFile = 'cuts.py' 

# file with list of samples
samplesFile = 'samples_DY_OS.py' 

# file with list of samples
plotFile = 'plot.py' 

# luminosity to normalize to (in 1/fb)
# https://github.com/latinos/LatinoAnalysis/blob/UL_production/NanoGardener/python/data/TrigMaker_cfg.py#L239 (#311 #377 #445)
# 0.418771191 + 7.653261227 + 7.866107374 + 0.8740119304 = 16.8121517224
lumi = 16.81

# used by mkPlot to define output directory for plots
# different from "outputDir" to do things more tidy
outputDirPlots = 'plots_'+tag

# used by mkDatacards to define output directory for datacards
outputDirDatacard = 'datacards'

# structure file for datacard
structureFile = 'structure.py'

# nuisances file for mkDatacards and for mkShape
nuisancesFile = 'nuisances_control_plots.py'
