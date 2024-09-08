# Configuration file to get reco-level nuisance normalizations
# PU : normalize at WW preselection level, only need to run once for all differential
# QCDscale_top : normalize for each top CR + corresponding SRs, only need to run once per choice of top CR binning
#                Currently top CR is binned either in number of total jets, or number of jets with |eta| < 2.5

treeName = 'Events'

tag = 'WW2016_noHIPM_v9_ExpNorm'

# used by mkShape to define output directory for root files
outputDir = 'rootFile'

# file with TTree aliases
aliasesFile = 'aliases.py'

# file with list of variables
variablesFile = 'variables.py'

# file with list of cuts
cutsFile = 'cuts.py' 

# file with list of samples
samplesFile = 'samples.py' 

# luminosity to normalize to (in 1/fb)
lumi = 16.81

# nuisances file for mkDatacards and for mkShape
nuisancesFile = 'nuisances.py'
