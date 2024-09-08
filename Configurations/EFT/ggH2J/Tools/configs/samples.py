import os
import inspect

configurations = os.path.realpath(inspect.getfile(inspect.currentframe())) # this file
configurations = os.path.dirname(configurations) # 
configurations = os.path.dirname(configurations) # 
configurations = os.path.dirname(configurations) # 
configurations = os.path.dirname(configurations) # Configurations

from LatinoAnalysis.Tools.commonTools import getSampleFiles, getBaseW, addSampleWeight

def nanoGetSampleFiles(inputDir, sample):
    try:
        if _samples_noload:
            return []
    except NameError:
        pass

    return getSampleFiles(inputDir, sample, True, 'nanoLatino_')

# samples

try:
    len(samples)
except NameError:
    import collections
    samples = collections.OrderedDict()

################################################
################# SKIMS ########################
################################################

mcProduction = 'Summer16_102X_nAODv7_Full2016v7'

dataReco = 'Run2016_102X_nAODv7_Full2016v7'

embedReco = 'Embedding2016_102X_nAODv7_Full2016v7'

mcSteps = 'MCl1loose2016v7__MCCorr2016v7__l2loose__l2tightOR2016v7{var}'

fakeSteps = 'DATAl1loose2016v7__l2loose__fakeW'

dataSteps = 'DATAl1loose2016v7__l2loose__l2tightOR2016v7'

embedSteps = 'DATAl1loose2016v7__l2loose__l2tightOR2016v7__Embedding'

##############################################
###### Tree base directory for the site ######
##############################################

SITE=os.uname()[1]
if    'iihe' in SITE:
  treeBaseDir = '/pnfs/iihe/cms/store/user/xjanssen/HWW2015'
elif  'cern' in SITE:
  treeBaseDir = '/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano'

def makeMCDirectory(var=''):
    if var:
        return os.path.join(treeBaseDir, mcProduction, mcSteps.format(var='__' + var))
    else:
        return os.path.join(treeBaseDir, mcProduction, mcSteps.format(var=''))

mcDirectory = makeMCDirectory()
fakeDirectory = os.path.join(treeBaseDir, dataReco, fakeSteps)
dataDirectory = os.path.join(treeBaseDir, dataReco, dataSteps)
embedDirectory = os.path.join(treeBaseDir, embedReco, embedSteps)

################################################
############ DATA DECLARATION ##################
################################################

DataRun = [
    ['B','Run2016B-02Apr2020_ver1-v1'],
    ['B','Run2016B-02Apr2020_ver2-v1'],
    ['C','Run2016C-02Apr2020-v1'],
    ['D','Run2016D-02Apr2020-v1'],
    ['E','Run2016E-02Apr2020-v1'],
    ['F','Run2016F-02Apr2020-v1'],
    ['G','Run2016G-02Apr2020-v1'],
    ['H','Run2016H-02Apr2020-v1']
]

DataSets = ['MuonEG','SingleMuon','SingleElectron','DoubleMuon', 'DoubleEG']

DataTrig = {
    'MuonEG'         : ' Trigger_ElMu' ,
    'SingleMuon'     : '!Trigger_ElMu && Trigger_sngMu' ,
    'SingleElectron' : '!Trigger_ElMu && !Trigger_sngMu && Trigger_sngEl',
    'DoubleMuon'     : '!Trigger_ElMu && !Trigger_sngMu && !Trigger_sngEl && Trigger_dblMu',
    'DoubleEG'       : '!Trigger_ElMu && !Trigger_sngMu && !Trigger_sngEl && !Trigger_dblMu && Trigger_dblEl'
}

#########################################
############ MC COMMON ##################
#########################################

# SFweight does not include btag weights
mcCommonWeightNoMatch = 'XSWeight*SFweight*METFilter_MC'
mcCommonWeight = 'XSWeight*SFweight*PromptGenLepMatch2l*METFilter_MC'

###########################################
#############  BACKGROUNDS  ###############
###########################################

###### DY #######

useEmbeddedDY = True
useDYtt = True

# The Dyveto sample is used to estimate one piece of the Dyemb uncertainty
# To avoid running it all the times, it was run once and the uncertainty was converted into a lnN (see nuisances.py)
runDYveto = True

embed_tautauveto = '' #Setup
if useEmbeddedDY:
  embed_tautauveto = '*embed_tautauveto'

if useEmbeddedDY:
  # Actual embedded data
  samples['Dyemb'] = {
    'name': [],
    'weight': 'METFilter_DATA*LepWPCut*Muon_ttHMVA_SF*embedtotal*genWeight*(genWeight<=1)',
    'weights': [],
    'isData': ['all'],
    'FilesPerJob': 20
  }

  for run_, sd in DataRun:
      files = nanoGetSampleFiles(embedDirectory, 'DYToTT_MuEle_Embedded_Run2016' + run_)
      samples['Dyemb']['name'].extend(files)
      samples['Dyemb']['weights'].extend(['Trigger_ElMu'] * len(files))

  # Vetoed MC: Needed for uncertainty
  files = nanoGetSampleFiles(mcDirectory, 'TTTo2L2Nu') + \
      nanoGetSampleFiles(mcDirectory, 'ST_tW_antitop') + \
      nanoGetSampleFiles(mcDirectory, 'ST_tW_top') + \
      nanoGetSampleFiles(mcDirectory, 'WWTo2L2Nu') + \
      nanoGetSampleFiles(mcDirectory, 'WpWmJJ_EWK_noTop') + \
      nanoGetSampleFiles(mcDirectory, 'GluGluWWTo2L2Nu_MCFM') + \
      nanoGetSampleFiles(mcDirectory, 'ZZTo2L2Nu') + \
      nanoGetSampleFiles(mcDirectory, 'ZZTo2L2Q') + \
      nanoGetSampleFiles(mcDirectory, 'ZZTo4L') + \
      nanoGetSampleFiles(mcDirectory, 'WZTo2L2Q') + \
      nanoGetSampleFiles(mcDirectory, 'Zg') + \
      nanoGetSampleFiles(mcDirectory, 'WZTo3LNu_mllmin01')

'''
  if runDYveto:
      samples['Dyveto'] = {
          'name': files,
          'weight': '(1-embed_tautauveto)',
          'FilesPerJob': 1, # There's some error about not finding sample-specific variables like "nllW" when mixing different samples into a single job; so split them all up instead
      }
    
      addSampleWeight(samples, 'Dyveto', 'TTTo2L2Nu', mcCommonWeight + '*((topGenPt * antitopGenPt > 0.) * (TMath::Sqrt((0.103*TMath::Exp(-0.0118*topGenPt) - 0.000134*topGenPt + 0.973) * (0.103*TMath::Exp(-0.0118*antitopGenPt) - 0.000134*antitopGenPt + 0.973))) * (TMath::Sqrt(TMath::Exp(1.61468e-03 + 3.46659e-06*topGenPt - 8.90557e-08*topGenPt*topGenPt) * TMath::Exp(1.61468e-03 + 3.46659e-06*antitopGenPt - 8.90557e-08*antitopGenPt*antitopGenPt))) + (topGenPt * antitopGenPt <= 0.))'), # Same Reweighting as other years, but with additional fix for tune CUET -> CP5)')
      addSampleWeight(samples, 'Dyveto', 'ST_tW_antitop', mcCommonWeight)
      addSampleWeight(samples, 'Dyveto', 'ST_tW_top', mcCommonWeight)
      addSampleWeight(samples, 'Dyveto', 'WWTo2L2Nu', mcCommonWeight + '*nllW')
      addSampleWeight(samples, 'Dyveto', 'WpWmJJ_EWK_noTop', mcCommonWeight + '*((Sum$(abs(GenPart_pdgId)==6 || GenPart_pdgId==25)==0)*(lhe_mWm > 60. && lhe_mWm < 100. && lhe_mWp > 60. && lhe_mWp < 100.))')
      addSampleWeight(samples, 'Dyveto', 'GluGluWWTo2L2Nu_MCFM', mcCommonWeight + '*1.53/1.4')
      addSampleWeight(samples, 'Dyveto', 'ZZTo2L2Nu', mcCommonWeight + '*1.11')
      addSampleWeight(samples, 'Dyveto', 'ZZTo2L2Q', mcCommonWeight + '*1.11')
      addSampleWeight(samples, 'Dyveto', 'ZZTo4L', mcCommonWeight + '*1.11')
      addSampleWeight(samples, 'Dyveto', 'WZTo2L2Q', mcCommonWeight + '*1.11')
      addSampleWeight(samples, 'Dyveto', 'Zg', ' ( ' + mcCommonWeightNoMatch + '*(!(Gen_ZGstar_mass > 0))' + ' ) + ( ' + mcCommonWeight + ' * ((Gen_ZGstar_mass >0 && Gen_ZGstar_mass < 4) * 0.94 + (Gen_ZGstar_mass <0 || Gen_ZGstar_mass > 4) * 1.14) * (Gen_ZGstar_mass > 0)' + ' ) ') # Vg contribution + VgS contribution
      addSampleWeight(samples, 'Dyveto', 'WZTo3LNu_mllmin01', mcCommonWeight + '*((Gen_ZGstar_mass >0 && Gen_ZGstar_mass < 4) * 0.94 + (Gen_ZGstar_mass <0 || Gen_ZGstar_mass > 4) * 1.14) * (Gen_ZGstar_mass > 0.1)')
'''

###### DY MC ######
## We need to keep DY MC as well, because only embedded events passing the ElMu trigger are considered
## Events failing ElMu but passing one of the other triggers are included in the DY MC
if useDYtt:
    files = nanoGetSampleFiles(mcDirectory, 'DYJetsToTT_MuEle_M-50_ext1') + \
        nanoGetSampleFiles(mcDirectory, 'DYJetsToLL_M-10to50-LO')
    
    samples['DY'] = {
        'name': files,
        'weight': mcCommonWeight+embed_tautauveto + '*( !(Sum$(PhotonGen_isPrompt==1 && PhotonGen_pt>15 && abs(PhotonGen_eta)<2.6) > 0))',
        'FilesPerJob': 2,
    }
    addSampleWeight(samples,'DY','DYJetsToTT_MuEle_M-50_ext1','DY_NLO_pTllrw')
    addSampleWeight(samples,'DY','DYJetsToLL_M-10to50-LO','DY_LO_pTllrw')
else:
    files = nanoGetSampleFiles(mcDirectory, 'DYJetsToLL_M-50') + \
        nanoGetSampleFiles(mcDirectory, 'DYJetsToLL_M-10to50-LO')

    samples['DY'] = {
        'name': files,
        'weight': mcCommonWeight+embed_tautauveto + "*( !(Sum$(PhotonGen_isPrompt==1 && PhotonGen_pt>15 && abs(PhotonGen_eta)<2.6) > 0 &&\
                                         Sum$(LeptonGen_isPrompt==1 && LeptonGen_pt>15)>=2) )",
        'FilesPerJob': 8,
    }
    addSampleWeight(samples,'DY','DYJetsToLL_M-50','DY_NLO_pTllrw')
    addSampleWeight(samples,'DY','DYJetsToLL_M-10to50-LO_ext1','DY_LO_pTllrw')


###### Top #######

files = nanoGetSampleFiles(mcDirectory, 'TTTo2L2Nu') + \
    nanoGetSampleFiles(mcDirectory, 'ST_s-channel') + \
    nanoGetSampleFiles(mcDirectory, 'ST_t-channel_antitop') + \
    nanoGetSampleFiles(mcDirectory, 'ST_t-channel_top') + \
    nanoGetSampleFiles(mcDirectory, 'ST_tW_antitop') + \
    nanoGetSampleFiles(mcDirectory, 'ST_tW_top')

samples['top'] = {
    'name': files,
    'weight': mcCommonWeight+embed_tautauveto,
    'FilesPerJob': 1,
}

addSampleWeight(samples,'top','TTTo2L2Nu','Top_pTrw')

###### WW ########

samples['WW'] = {
    'name': nanoGetSampleFiles(mcDirectory, 'WWTo2L2Nu'),
    'weight': mcCommonWeight+embed_tautauveto + '*nllW', # temporary - nllW module not run on PS and UE variation samples
    'FilesPerJob': 1
}


samples['WWewk'] = {
    'name': nanoGetSampleFiles(mcDirectory, 'WpWmJJ_EWK_noTop'),
    'weight': mcCommonWeight+embed_tautauveto + '*(Sum$(abs(GenPart_pdgId)==6 || GenPart_pdgId==25)==0)*(lhe_mWm > 60. && lhe_mWm < 100. && lhe_mWp > 60. && lhe_mWp < 100.)', #filter tops and Higgs, limit w mass
    'FilesPerJob': 4
}

samples['ggWW'] = {
    'name': nanoGetSampleFiles(mcDirectory, 'GluGluWWTo2L2Nu_MCFM'),
    'weight': mcCommonWeight+embed_tautauveto + '*1.53/1.4', # updating k-factor
    'FilesPerJob': 4
}

######## Vg ########

files = nanoGetSampleFiles(mcDirectory, 'Wg_MADGRAPHMLM') + \
    nanoGetSampleFiles(mcDirectory, 'Zg')

samples['Vg'] = {
    'name': files,
    'weight': mcCommonWeightNoMatch+embed_tautauveto + '*(!(Gen_ZGstar_mass > 0))',
    'FilesPerJob': 4
}

######## VgS ########

files = nanoGetSampleFiles(mcDirectory, 'Wg_MADGRAPHMLM') + \
    nanoGetSampleFiles(mcDirectory, 'Zg') + \
    nanoGetSampleFiles(mcDirectory, 'WZTo3LNu_mllmin01')

samples['VgS'] = {
    'name': files,
    'weight': mcCommonWeight+embed_tautauveto + ' * (gstarLow * 0.94 + gstarHigh * 1.14)',
    'FilesPerJob': 4,
    'subsamples': {
      'L': 'gstarLow',
      'H': 'gstarHigh'
    }
}
addSampleWeight(samples, 'VgS', 'Wg_MADGRAPHMLM', '(Gen_ZGstar_mass > 0 && Gen_ZGstar_mass < 0.1)')
addSampleWeight(samples, 'VgS', 'Zg', '(Gen_ZGstar_mass > 0)')
addSampleWeight(samples, 'VgS', 'WZTo3LNu_mllmin01', '(Gen_ZGstar_mass > 0.1)')

############ VZ ############

files = nanoGetSampleFiles(mcDirectory, 'ZZTo2L2Nu') + \
    nanoGetSampleFiles(mcDirectory, 'ZZTo2L2Q') + \
    nanoGetSampleFiles(mcDirectory, 'ZZTo4L') + \
    nanoGetSampleFiles(mcDirectory, 'WZTo2L2Q')

samples['VZ'] = {
    'name': files,
    'weight': mcCommonWeight+embed_tautauveto + '*1.11',
    'FilesPerJob': 4
}

########## VVV #########

files = nanoGetSampleFiles(mcDirectory, 'ZZZ') + \
    nanoGetSampleFiles(mcDirectory, 'WZZ') + \
    nanoGetSampleFiles(mcDirectory, 'WWZ') + \
    nanoGetSampleFiles(mcDirectory, 'WWW')
#+ nanoGetSampleFiles(mcDirectory, 'WWG'), #should this be included? or is it already taken into account in the WW sample?

samples['VVV'] = {
    'name': files,
    'weight': mcCommonWeight,
    'FilesPerJob': 4
}

###########################################
#############   SIGNALS  ##################
###########################################

signals = []

#### ggH -> WW
samples['ggH_hww'] = {
    'name': nanoGetSampleFiles(mcDirectory, 'GluGluHToWWTo2L2Nu_alternative_M125')+nanoGetSampleFiles(mcDirectory, 'GGHjjToWWTo2L2Nu_minloHJJ_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 1,
}
addSampleWeight(samples, 'ggH_hww', 'GluGluHToWWTo2L2Nu_alternative_M125', '(HTXS_stage1_1_cat_pTjet30GeV<107)*Weight2MINLO*1092.0713/1068.1909') #only non GE2J categories with the weight to NNLOPS and renormalize integral                          
addSampleWeight(samples, 'ggH_hww', 'GGHjjToWWTo2L2Nu_minloHJJ_M125', '(HTXS_stage1_1_cat_pTjet30GeV>106)*1092.0713/1068.1909')

signals.append('ggH_hww')

############ VBF H->WW ############
samples['qqH_hww'] = {
    'name': nanoGetSampleFiles(mcDirectory, 'VBFHToWWTo2L2Nu_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 4
}

signals.append('qqH_hww')

############ ZH H->WW ############

samples['ZH_hww'] = {
    'name':   nanoGetSampleFiles(mcDirectory, 'HZJ_HToWW_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 4
}

signals.append('ZH_hww')

samples['ggZH_hww'] = {
    'name':   nanoGetSampleFiles(mcDirectory, 'ggZH_HToWW_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 4
}

signals.append('ggZH_hww')

############ WH H->WW ############

samples['WH_hww'] = {
    'name':   nanoGetSampleFiles(mcDirectory, 'HWplusJ_HToWW_M125') + nanoGetSampleFiles(mcDirectory, 'HWminusJ_HToWW_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 4
}

signals.append('WH_hww')

############ ttH ############

samples['ttH_hww'] = {
    'name':   nanoGetSampleFiles(mcDirectory, 'ttHToNonbb_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 5
}

signals.append('ttH_hww')

############ H->TauTau ############

samples['ggH_htt'] = {
    'name': nanoGetSampleFiles(mcDirectory, 'GluGluHToTauTau_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 4
}

signals.append('ggH_htt')

samples['qqH_htt'] = {
    'name': nanoGetSampleFiles(mcDirectory, 'VBFHToTauTau_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 4
}
signals.append('qqH_htt')

samples['ZH_htt'] = {
    'name': nanoGetSampleFiles(mcDirectory, 'HZJ_HToTauTau_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 4
}

signals.append('ZH_htt')

samples['WH_htt'] = {
    'name':  nanoGetSampleFiles(mcDirectory, 'HWplusJ_HToTauTau_M125') + nanoGetSampleFiles(mcDirectory, 'HWminusJ_HToTauTau_M125'),
    'weight': mcCommonWeight,
    'FilesPerJob': 4
}

signals.append('WH_htt')


###########################################
################## FAKE ###################
###########################################

samples['Fake'] = {
  'name': [],
  'weight': 'METFilter_DATA*fakeW',
  'weights': [],
  'isData': ['all'],
  'FilesPerJob': 50
}

for _, sd in DataRun:
  for pd in DataSets:
    files = nanoGetSampleFiles(fakeDirectory, pd + '_' + sd)

    samples['Fake']['name'].extend(files)
    samples['Fake']['weights'].extend([DataTrig[pd]] * len(files))

samples['Fake']['subsamples'] = {
  'em': 'abs(Lepton_pdgId[0]) == 11',
  'me': 'abs(Lepton_pdgId[0]) == 13'
}

###########################################
################## DATA ###################
###########################################

samples['DATA'] = {
  'name': [],
  'weight': 'METFilter_DATA*LepWPCut',
  'weights': [],
  'isData': ['all'],
  'FilesPerJob': 50
}

for _, sd in DataRun:
  for pd in DataSets:
    files = nanoGetSampleFiles(dataDirectory, pd + '_' + sd)
    
    samples['DATA']['name'].extend(files)
    samples['DATA']['weights'].extend([DataTrig[pd]] * len(files))


  

 
#### AC/EFT Signals 
 
signals_rw = [] 
 
# VBF MC samples 
 
# Original VBF samples 
 
samples['VBF_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W',   'FilesPerJob': 4, } 
 
samples['VBF_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W',   'FilesPerJob': 4, } 
 
samples['VBF_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W',   'FilesPerJob': 4, } 
 
samples['VBF_H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W',   'FilesPerJob': 4, } 
 
samples['VBF_H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W',   'FilesPerJob': 4, } 
 
samples['VBF_H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W',   'FilesPerJob': 4, } 
 
samples['VBF_H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W',   'FilesPerJob': 4, } 
 
# Reweighted VBF samples 
 
samples['VBF_H0PM_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0M_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0M_M0')  
 
samples['VBF_H0PM_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0M_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0M_M1')  
 
samples['VBF_H0PM_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0M_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0M_M2')  
 
samples['VBF_H0PM_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0M_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0M_M3')  
 
samples['VBF_H0PM_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0PH_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0PH_M0')  
 
samples['VBF_H0PM_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0PH_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0PH_M1')  
 
samples['VBF_H0PM_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0PH_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0PH_M2')  
 
samples['VBF_H0PM_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0PH_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0PH_M3')  
 
samples['VBF_H0PM_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0L1_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0L1_M0')  
 
samples['VBF_H0PM_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0L1_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0L1_M1')  
 
samples['VBF_H0PM_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0L1_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0L1_M2')  
 
samples['VBF_H0PM_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0L1_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0L1_M3')  
 
samples['VBF_H0PM_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0LZg_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0LZg_M0')  
 
samples['VBF_H0PM_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0LZg_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0LZg_M1')  
 
samples['VBF_H0PM_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0LZg_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0LZg_M2')  
 
samples['VBF_H0PM_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_H0LZg_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_H0LZg_M3')  
 
samples['VBF_H0PM_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0M_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0M_M0')  
 
samples['VBF_H0PM_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0M_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0M_M1')  
 
samples['VBF_H0PM_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0M_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0M_M2')  
 
samples['VBF_H0PM_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0M_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0M_M3')  
 
samples['VBF_H0PM_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0PH_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0PH_M0')  
 
samples['VBF_H0PM_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0PH_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0PH_M1')  
 
samples['VBF_H0PM_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0PH_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0PH_M2')  
 
samples['VBF_H0PM_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0PH_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0PH_M3')  
 
samples['VBF_H0PM_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0L1_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0L1_M0')  
 
samples['VBF_H0PM_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0L1_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0L1_M1')  
 
samples['VBF_H0PM_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0L1_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0L1_M2')  
 
samples['VBF_H0PM_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PM_W*(ME_EFTH0L1_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PM_EFTH0L1_M3')  
 
samples['VBF_H0M_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0PM/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0PM')  
 
samples['VBF_H0M_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0M_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0M_M0')  
 
samples['VBF_H0M_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0M_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0M_M1')  
 
samples['VBF_H0M_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0M_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0M_M2')  
 
samples['VBF_H0M_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0M_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0M_M3')  
 
samples['VBF_H0M_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0PH_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0PH_M0')  
 
samples['VBF_H0M_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0PH_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0PH_M1')  
 
samples['VBF_H0M_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0PH_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0PH_M2')  
 
samples['VBF_H0M_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0PH_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0PH_M3')  
 
samples['VBF_H0M_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0L1_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0L1_M0')  
 
samples['VBF_H0M_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0L1_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0L1_M1')  
 
samples['VBF_H0M_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0L1_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0L1_M2')  
 
samples['VBF_H0M_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0L1_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0L1_M3')  
 
samples['VBF_H0M_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0LZg_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0LZg_M0')  
 
samples['VBF_H0M_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0LZg_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0LZg_M1')  
 
samples['VBF_H0M_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0LZg_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0LZg_M2')  
 
samples['VBF_H0M_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_H0LZg_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_H0LZg_M3')  
 
samples['VBF_H0M_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0M_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0M_M0')  
 
samples['VBF_H0M_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0M_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0M_M1')  
 
samples['VBF_H0M_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0M_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0M_M2')  
 
samples['VBF_H0M_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0M_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0M_M3')  
 
samples['VBF_H0M_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0PH_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0PH_M0')  
 
samples['VBF_H0M_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0PH_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0PH_M1')  
 
samples['VBF_H0M_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0PH_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0PH_M2')  
 
samples['VBF_H0M_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0PH_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0PH_M3')  
 
samples['VBF_H0M_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0L1_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0L1_M0')  
 
samples['VBF_H0M_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0L1_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0L1_M1')  
 
samples['VBF_H0M_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0L1_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0L1_M2')  
 
samples['VBF_H0M_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0M_W*(ME_EFTH0L1_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0M_EFTH0L1_M3')  
 
samples['VBF_H0Mf05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0PM/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0PM')  
 
samples['VBF_H0Mf05_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0M_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0M_M0')  
 
samples['VBF_H0Mf05_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0M_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0M_M1')  
 
samples['VBF_H0Mf05_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0M_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0M_M2')  
 
samples['VBF_H0Mf05_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0M_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0M_M3')  
 
samples['VBF_H0Mf05_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0PH_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0PH_M0')  
 
samples['VBF_H0Mf05_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0PH_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0PH_M1')  
 
samples['VBF_H0Mf05_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0PH_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0PH_M2')  
 
samples['VBF_H0Mf05_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0PH_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0PH_M3')  
 
samples['VBF_H0Mf05_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0L1_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0L1_M0')  
 
samples['VBF_H0Mf05_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0L1_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0L1_M1')  
 
samples['VBF_H0Mf05_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0L1_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0L1_M2')  
 
samples['VBF_H0Mf05_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0L1_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0L1_M3')  
 
samples['VBF_H0Mf05_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0LZg_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0LZg_M0')  
 
samples['VBF_H0Mf05_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0LZg_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0LZg_M1')  
 
samples['VBF_H0Mf05_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0LZg_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0LZg_M2')  
 
samples['VBF_H0Mf05_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_H0LZg_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_H0LZg_M3')  
 
samples['VBF_H0Mf05_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0M_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0M_M0')  
 
samples['VBF_H0Mf05_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0M_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0M_M1')  
 
samples['VBF_H0Mf05_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0M_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0M_M2')  
 
samples['VBF_H0Mf05_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0M_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0M_M3')  
 
samples['VBF_H0Mf05_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0PH_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0PH_M0')  
 
samples['VBF_H0Mf05_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0PH_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0PH_M1')  
 
samples['VBF_H0Mf05_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0PH_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0PH_M2')  
 
samples['VBF_H0Mf05_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0PH_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0PH_M3')  
 
samples['VBF_H0Mf05_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0L1_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0L1_M0')  
 
samples['VBF_H0Mf05_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0L1_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0L1_M1')  
 
samples['VBF_H0Mf05_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0L1_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0L1_M2')  
 
samples['VBF_H0Mf05_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0Mf05_W*(ME_EFTH0L1_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0Mf05_EFTH0L1_M3')  
 
samples['VBF_H0PH_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0PM/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0PM')  
 
samples['VBF_H0PH_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0M_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0M_M0')  
 
samples['VBF_H0PH_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0M_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0M_M1')  
 
samples['VBF_H0PH_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0M_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0M_M2')  
 
samples['VBF_H0PH_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0M_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0M_M3')  
 
samples['VBF_H0PH_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0PH_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0PH_M0')  
 
samples['VBF_H0PH_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0PH_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0PH_M1')  
 
samples['VBF_H0PH_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0PH_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0PH_M2')  
 
samples['VBF_H0PH_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0PH_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0PH_M3')  
 
samples['VBF_H0PH_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0L1_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0L1_M0')  
 
samples['VBF_H0PH_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0L1_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0L1_M1')  
 
samples['VBF_H0PH_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0L1_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0L1_M2')  
 
samples['VBF_H0PH_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0L1_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0L1_M3')  
 
samples['VBF_H0PH_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0LZg_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0LZg_M0')  
 
samples['VBF_H0PH_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0LZg_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0LZg_M1')  
 
samples['VBF_H0PH_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0LZg_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0LZg_M2')  
 
samples['VBF_H0PH_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_H0LZg_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_H0LZg_M3')  
 
samples['VBF_H0PH_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0M_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0M_M0')  
 
samples['VBF_H0PH_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0M_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0M_M1')  
 
samples['VBF_H0PH_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0M_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0M_M2')  
 
samples['VBF_H0PH_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0M_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0M_M3')  
 
samples['VBF_H0PH_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0PH_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0PH_M0')  
 
samples['VBF_H0PH_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0PH_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0PH_M1')  
 
samples['VBF_H0PH_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0PH_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0PH_M2')  
 
samples['VBF_H0PH_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0PH_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0PH_M3')  
 
samples['VBF_H0PH_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0L1_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0L1_M0')  
 
samples['VBF_H0PH_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0L1_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0L1_M1')  
 
samples['VBF_H0PH_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0L1_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0L1_M2')  
 
samples['VBF_H0PH_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PH_W*(ME_EFTH0L1_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PH_EFTH0L1_M3')  
 
samples['VBF_H0PHf05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0PM/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0PM')  
 
samples['VBF_H0PHf05_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0M_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0M_M0')  
 
samples['VBF_H0PHf05_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0M_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0M_M1')  
 
samples['VBF_H0PHf05_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0M_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0M_M2')  
 
samples['VBF_H0PHf05_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0M_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0M_M3')  
 
samples['VBF_H0PHf05_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0PH_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0PH_M0')  
 
samples['VBF_H0PHf05_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0PH_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0PH_M1')  
 
samples['VBF_H0PHf05_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0PH_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0PH_M2')  
 
samples['VBF_H0PHf05_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0PH_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0PH_M3')  
 
samples['VBF_H0PHf05_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0L1_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0L1_M0')  
 
samples['VBF_H0PHf05_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0L1_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0L1_M1')  
 
samples['VBF_H0PHf05_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0L1_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0L1_M2')  
 
samples['VBF_H0PHf05_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0L1_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0L1_M3')  
 
samples['VBF_H0PHf05_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0LZg_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0LZg_M0')  
 
samples['VBF_H0PHf05_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0LZg_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0LZg_M1')  
 
samples['VBF_H0PHf05_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0LZg_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0LZg_M2')  
 
samples['VBF_H0PHf05_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_H0LZg_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_H0LZg_M3')  
 
samples['VBF_H0PHf05_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0M_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0M_M0')  
 
samples['VBF_H0PHf05_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0M_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0M_M1')  
 
samples['VBF_H0PHf05_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0M_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0M_M2')  
 
samples['VBF_H0PHf05_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0M_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0M_M3')  
 
samples['VBF_H0PHf05_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0PH_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0PH_M0')  
 
samples['VBF_H0PHf05_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0PH_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0PH_M1')  
 
samples['VBF_H0PHf05_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0PH_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0PH_M2')  
 
samples['VBF_H0PHf05_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0PH_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0PH_M3')  
 
samples['VBF_H0PHf05_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0L1_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0L1_M0')  
 
samples['VBF_H0PHf05_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0L1_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0L1_M1')  
 
samples['VBF_H0PHf05_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0L1_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0L1_M2')  
 
samples['VBF_H0PHf05_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0PHf05_W*(ME_EFTH0L1_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0PHf05_EFTH0L1_M3')  
 
samples['VBF_H0L1_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0PM/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0PM')  
 
samples['VBF_H0L1_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0M_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0M_M0')  
 
samples['VBF_H0L1_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0M_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0M_M1')  
 
samples['VBF_H0L1_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0M_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0M_M2')  
 
samples['VBF_H0L1_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0M_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0M_M3')  
 
samples['VBF_H0L1_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0PH_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0PH_M0')  
 
samples['VBF_H0L1_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0PH_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0PH_M1')  
 
samples['VBF_H0L1_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0PH_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0PH_M2')  
 
samples['VBF_H0L1_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0PH_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0PH_M3')  
 
samples['VBF_H0L1_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0L1_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0L1_M0')  
 
samples['VBF_H0L1_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0L1_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0L1_M1')  
 
samples['VBF_H0L1_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0L1_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0L1_M2')  
 
samples['VBF_H0L1_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0L1_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0L1_M3')  
 
samples['VBF_H0L1_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0LZg_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0LZg_M0')  
 
samples['VBF_H0L1_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0LZg_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0LZg_M1')  
 
samples['VBF_H0L1_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0LZg_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0LZg_M2')  
 
samples['VBF_H0L1_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_H0LZg_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_H0LZg_M3')  
 
samples['VBF_H0L1_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0M_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0M_M0')  
 
samples['VBF_H0L1_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0M_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0M_M1')  
 
samples['VBF_H0L1_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0M_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0M_M2')  
 
samples['VBF_H0L1_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0M_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0M_M3')  
 
samples['VBF_H0L1_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0PH_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0PH_M0')  
 
samples['VBF_H0L1_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0PH_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0PH_M1')  
 
samples['VBF_H0L1_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0PH_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0PH_M2')  
 
samples['VBF_H0L1_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0PH_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0PH_M3')  
 
samples['VBF_H0L1_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0L1_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0L1_M0')  
 
samples['VBF_H0L1_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0L1_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0L1_M1')  
 
samples['VBF_H0L1_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0L1_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0L1_M2')  
 
samples['VBF_H0L1_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1_W*(ME_EFTH0L1_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1_EFTH0L1_M3')  
 
samples['VBF_H0L1f05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0PM/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0PM')  
 
samples['VBF_H0L1f05_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0M_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0M_M0')  
 
samples['VBF_H0L1f05_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0M_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0M_M1')  
 
samples['VBF_H0L1f05_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0M_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0M_M2')  
 
samples['VBF_H0L1f05_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0M_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0M_M3')  
 
samples['VBF_H0L1f05_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0PH_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0PH_M0')  
 
samples['VBF_H0L1f05_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0PH_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0PH_M1')  
 
samples['VBF_H0L1f05_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0PH_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0PH_M2')  
 
samples['VBF_H0L1f05_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0PH_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0PH_M3')  
 
samples['VBF_H0L1f05_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0L1_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0L1_M0')  
 
samples['VBF_H0L1f05_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0L1_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0L1_M1')  
 
samples['VBF_H0L1f05_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0L1_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0L1_M2')  
 
samples['VBF_H0L1f05_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0L1_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0L1_M3')  
 
samples['VBF_H0L1f05_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0LZg_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0LZg_M0')  
 
samples['VBF_H0L1f05_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0LZg_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0LZg_M1')  
 
samples['VBF_H0L1f05_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0LZg_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0LZg_M2')  
 
samples['VBF_H0L1f05_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_H0LZg_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_H0LZg_M3')  
 
samples['VBF_H0L1f05_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0M_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0M_M0')  
 
samples['VBF_H0L1f05_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0M_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0M_M1')  
 
samples['VBF_H0L1f05_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0M_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0M_M2')  
 
samples['VBF_H0L1f05_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0M_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0M_M3')  
 
samples['VBF_H0L1f05_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0PH_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0PH_M0')  
 
samples['VBF_H0L1f05_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0PH_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0PH_M1')  
 
samples['VBF_H0L1f05_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0PH_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0PH_M2')  
 
samples['VBF_H0L1f05_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0PH_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0PH_M3')  
 
samples['VBF_H0L1f05_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0L1_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0L1_M0')  
 
samples['VBF_H0L1f05_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0L1_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0L1_M1')  
 
samples['VBF_H0L1f05_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0L1_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0L1_M2')  
 
samples['VBF_H0L1f05_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'VBF_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*VBF_H0L1f05_W*(ME_EFTH0L1_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('VBF_H0L1f05_EFTH0L1_M3')  
 
# WH MC samples 
 
# Original WH samples 
 
samples['WH_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W',   'FilesPerJob': 4, } 
 
samples['WH_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W',   'FilesPerJob': 4, } 
 
samples['WH_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W',   'FilesPerJob': 4, } 
 
samples['WH_H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W',   'FilesPerJob': 4, } 
 
samples['WH_H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W',   'FilesPerJob': 4, } 
 
samples['WH_H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W',   'FilesPerJob': 4, } 
 
samples['WH_H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W',   'FilesPerJob': 4, } 
 
# Reweighted WH samples 
 
samples['WH_H0PM_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0M_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0M_M0')  
 
samples['WH_H0PM_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0M_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0M_M1')  
 
samples['WH_H0PM_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0M_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0M_M2')  
 
samples['WH_H0PM_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0M_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0M_M3')  
 
samples['WH_H0PM_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0PH_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0PH_M0')  
 
samples['WH_H0PM_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0PH_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0PH_M1')  
 
samples['WH_H0PM_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0PH_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0PH_M2')  
 
samples['WH_H0PM_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0PH_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0PH_M3')  
 
samples['WH_H0PM_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0L1_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0L1_M0')  
 
samples['WH_H0PM_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0L1_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0L1_M1')  
 
samples['WH_H0PM_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0L1_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0L1_M2')  
 
samples['WH_H0PM_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_H0L1_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_H0L1_M3')  
 
samples['WH_H0PM_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0M_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0M_M0')  
 
samples['WH_H0PM_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0M_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0M_M1')  
 
samples['WH_H0PM_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0M_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0M_M2')  
 
samples['WH_H0PM_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0M_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0M_M3')  
 
samples['WH_H0PM_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0PH_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0PH_M0')  
 
samples['WH_H0PM_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0PH_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0PH_M1')  
 
samples['WH_H0PM_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0PH_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0PH_M2')  
 
samples['WH_H0PM_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0PH_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0PH_M3')  
 
samples['WH_H0PM_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0L1_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0L1_M0')  
 
samples['WH_H0PM_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0L1_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0L1_M1')  
 
samples['WH_H0PM_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0L1_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0L1_M2')  
 
samples['WH_H0PM_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PM_W*(ME_EFTH0L1_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PM_EFTH0L1_M3')  
 
samples['WH_H0M_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0PM/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0PM')  
 
samples['WH_H0M_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0M_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0M_M0')  
 
samples['WH_H0M_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0M_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0M_M1')  
 
samples['WH_H0M_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0M_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0M_M2')  
 
samples['WH_H0M_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0M_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0M_M3')  
 
samples['WH_H0M_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0PH_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0PH_M0')  
 
samples['WH_H0M_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0PH_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0PH_M1')  
 
samples['WH_H0M_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0PH_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0PH_M2')  
 
samples['WH_H0M_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0PH_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0PH_M3')  
 
samples['WH_H0M_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0L1_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0L1_M0')  
 
samples['WH_H0M_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0L1_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0L1_M1')  
 
samples['WH_H0M_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0L1_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0L1_M2')  
 
samples['WH_H0M_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_H0L1_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_H0L1_M3')  
 
samples['WH_H0M_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0M_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0M_M0')  
 
samples['WH_H0M_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0M_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0M_M1')  
 
samples['WH_H0M_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0M_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0M_M2')  
 
samples['WH_H0M_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0M_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0M_M3')  
 
samples['WH_H0M_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0PH_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0PH_M0')  
 
samples['WH_H0M_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0PH_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0PH_M1')  
 
samples['WH_H0M_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0PH_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0PH_M2')  
 
samples['WH_H0M_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0PH_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0PH_M3')  
 
samples['WH_H0M_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0L1_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0L1_M0')  
 
samples['WH_H0M_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0L1_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0L1_M1')  
 
samples['WH_H0M_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0L1_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0L1_M2')  
 
samples['WH_H0M_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0M_W*(ME_EFTH0L1_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0M_EFTH0L1_M3')  
 
samples['WH_H0Mf05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0PM/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0PM')  
 
samples['WH_H0Mf05_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0M_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0M_M0')  
 
samples['WH_H0Mf05_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0M_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0M_M1')  
 
samples['WH_H0Mf05_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0M_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0M_M2')  
 
samples['WH_H0Mf05_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0M_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0M_M3')  
 
samples['WH_H0Mf05_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0PH_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0PH_M0')  
 
samples['WH_H0Mf05_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0PH_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0PH_M1')  
 
samples['WH_H0Mf05_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0PH_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0PH_M2')  
 
samples['WH_H0Mf05_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0PH_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0PH_M3')  
 
samples['WH_H0Mf05_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0L1_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0L1_M0')  
 
samples['WH_H0Mf05_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0L1_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0L1_M1')  
 
samples['WH_H0Mf05_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0L1_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0L1_M2')  
 
samples['WH_H0Mf05_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_H0L1_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_H0L1_M3')  
 
samples['WH_H0Mf05_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0M_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0M_M0')  
 
samples['WH_H0Mf05_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0M_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0M_M1')  
 
samples['WH_H0Mf05_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0M_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0M_M2')  
 
samples['WH_H0Mf05_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0M_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0M_M3')  
 
samples['WH_H0Mf05_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0PH_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0PH_M0')  
 
samples['WH_H0Mf05_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0PH_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0PH_M1')  
 
samples['WH_H0Mf05_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0PH_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0PH_M2')  
 
samples['WH_H0Mf05_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0PH_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0PH_M3')  
 
samples['WH_H0Mf05_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0L1_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0L1_M0')  
 
samples['WH_H0Mf05_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0L1_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0L1_M1')  
 
samples['WH_H0Mf05_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0L1_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0L1_M2')  
 
samples['WH_H0Mf05_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0Mf05_W*(ME_EFTH0L1_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0Mf05_EFTH0L1_M3')  
 
samples['WH_H0PH_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0PM/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0PM')  
 
samples['WH_H0PH_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0M_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0M_M0')  
 
samples['WH_H0PH_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0M_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0M_M1')  
 
samples['WH_H0PH_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0M_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0M_M2')  
 
samples['WH_H0PH_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0M_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0M_M3')  
 
samples['WH_H0PH_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0PH_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0PH_M0')  
 
samples['WH_H0PH_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0PH_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0PH_M1')  
 
samples['WH_H0PH_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0PH_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0PH_M2')  
 
samples['WH_H0PH_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0PH_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0PH_M3')  
 
samples['WH_H0PH_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0L1_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0L1_M0')  
 
samples['WH_H0PH_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0L1_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0L1_M1')  
 
samples['WH_H0PH_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0L1_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0L1_M2')  
 
samples['WH_H0PH_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_H0L1_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_H0L1_M3')  
 
samples['WH_H0PH_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0M_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0M_M0')  
 
samples['WH_H0PH_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0M_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0M_M1')  
 
samples['WH_H0PH_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0M_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0M_M2')  
 
samples['WH_H0PH_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0M_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0M_M3')  
 
samples['WH_H0PH_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0PH_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0PH_M0')  
 
samples['WH_H0PH_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0PH_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0PH_M1')  
 
samples['WH_H0PH_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0PH_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0PH_M2')  
 
samples['WH_H0PH_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0PH_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0PH_M3')  
 
samples['WH_H0PH_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0L1_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0L1_M0')  
 
samples['WH_H0PH_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0L1_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0L1_M1')  
 
samples['WH_H0PH_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0L1_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0L1_M2')  
 
samples['WH_H0PH_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PH_W*(ME_EFTH0L1_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PH_EFTH0L1_M3')  
 
samples['WH_H0PHf05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0PM/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0PM')  
 
samples['WH_H0PHf05_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0M_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0M_M0')  
 
samples['WH_H0PHf05_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0M_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0M_M1')  
 
samples['WH_H0PHf05_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0M_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0M_M2')  
 
samples['WH_H0PHf05_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0M_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0M_M3')  
 
samples['WH_H0PHf05_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0PH_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0PH_M0')  
 
samples['WH_H0PHf05_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0PH_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0PH_M1')  
 
samples['WH_H0PHf05_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0PH_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0PH_M2')  
 
samples['WH_H0PHf05_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0PH_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0PH_M3')  
 
samples['WH_H0PHf05_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0L1_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0L1_M0')  
 
samples['WH_H0PHf05_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0L1_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0L1_M1')  
 
samples['WH_H0PHf05_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0L1_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0L1_M2')  
 
samples['WH_H0PHf05_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_H0L1_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_H0L1_M3')  
 
samples['WH_H0PHf05_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0M_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0M_M0')  
 
samples['WH_H0PHf05_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0M_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0M_M1')  
 
samples['WH_H0PHf05_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0M_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0M_M2')  
 
samples['WH_H0PHf05_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0M_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0M_M3')  
 
samples['WH_H0PHf05_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0PH_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0PH_M0')  
 
samples['WH_H0PHf05_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0PH_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0PH_M1')  
 
samples['WH_H0PHf05_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0PH_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0PH_M2')  
 
samples['WH_H0PHf05_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0PH_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0PH_M3')  
 
samples['WH_H0PHf05_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0L1_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0L1_M0')  
 
samples['WH_H0PHf05_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0L1_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0L1_M1')  
 
samples['WH_H0PHf05_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0L1_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0L1_M2')  
 
samples['WH_H0PHf05_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0PHf05_W*(ME_EFTH0L1_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0PHf05_EFTH0L1_M3')  
 
samples['WH_H0L1_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0PM/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0PM')  
 
samples['WH_H0L1_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0M_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0M_M0')  
 
samples['WH_H0L1_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0M_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0M_M1')  
 
samples['WH_H0L1_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0M_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0M_M2')  
 
samples['WH_H0L1_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0M_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0M_M3')  
 
samples['WH_H0L1_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0PH_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0PH_M0')  
 
samples['WH_H0L1_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0PH_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0PH_M1')  
 
samples['WH_H0L1_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0PH_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0PH_M2')  
 
samples['WH_H0L1_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0PH_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0PH_M3')  
 
samples['WH_H0L1_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0L1_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0L1_M0')  
 
samples['WH_H0L1_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0L1_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0L1_M1')  
 
samples['WH_H0L1_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0L1_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0L1_M2')  
 
samples['WH_H0L1_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_H0L1_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_H0L1_M3')  
 
samples['WH_H0L1_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0M_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0M_M0')  
 
samples['WH_H0L1_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0M_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0M_M1')  
 
samples['WH_H0L1_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0M_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0M_M2')  
 
samples['WH_H0L1_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0M_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0M_M3')  
 
samples['WH_H0L1_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0PH_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0PH_M0')  
 
samples['WH_H0L1_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0PH_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0PH_M1')  
 
samples['WH_H0L1_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0PH_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0PH_M2')  
 
samples['WH_H0L1_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0PH_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0PH_M3')  
 
samples['WH_H0L1_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0L1_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0L1_M0')  
 
samples['WH_H0L1_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0L1_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0L1_M1')  
 
samples['WH_H0L1_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0L1_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0L1_M2')  
 
samples['WH_H0L1_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1_W*(ME_EFTH0L1_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1_EFTH0L1_M3')  
 
samples['WH_H0L1f05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0PM/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0PM')  
 
samples['WH_H0L1f05_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0M_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0M_M0')  
 
samples['WH_H0L1f05_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0M_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0M_M1')  
 
samples['WH_H0L1f05_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0M_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0M_M2')  
 
samples['WH_H0L1f05_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0M_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0M_M3')  
 
samples['WH_H0L1f05_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0PH_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0PH_M0')  
 
samples['WH_H0L1f05_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0PH_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0PH_M1')  
 
samples['WH_H0L1f05_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0PH_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0PH_M2')  
 
samples['WH_H0L1f05_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0PH_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0PH_M3')  
 
samples['WH_H0L1f05_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0L1_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0L1_M0')  
 
samples['WH_H0L1f05_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0L1_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0L1_M1')  
 
samples['WH_H0L1f05_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0L1_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0L1_M2')  
 
samples['WH_H0L1f05_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_H0L1_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_H0L1_M3')  
 
samples['WH_H0L1f05_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0M_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0M_M0')  
 
samples['WH_H0L1f05_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0M_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0M_M1')  
 
samples['WH_H0L1f05_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0M_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0M_M2')  
 
samples['WH_H0L1f05_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0M_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0M_M3')  
 
samples['WH_H0L1f05_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0PH_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0PH_M0')  
 
samples['WH_H0L1f05_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0PH_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0PH_M1')  
 
samples['WH_H0L1f05_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0PH_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0PH_M2')  
 
samples['WH_H0L1f05_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0PH_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0PH_M3')  
 
samples['WH_H0L1f05_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0L1_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0L1_M0')  
 
samples['WH_H0L1f05_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0L1_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0L1_M1')  
 
samples['WH_H0L1f05_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0L1_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0L1_M2')  
 
samples['WH_H0L1f05_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'WH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*WH_H0L1f05_W*(ME_EFTH0L1_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('WH_H0L1f05_EFTH0L1_M3')  
 
# ZH MC samples 
 
# Original ZH samples 
 
samples['ZH_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W',   'FilesPerJob': 4, } 
 
samples['ZH_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W',   'FilesPerJob': 4, } 
 
samples['ZH_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W',   'FilesPerJob': 4, } 
 
samples['ZH_H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W',   'FilesPerJob': 4, } 
 
samples['ZH_H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W',   'FilesPerJob': 4, } 
 
samples['ZH_H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W',   'FilesPerJob': 4, } 
 
samples['ZH_H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W',   'FilesPerJob': 4, } 
 
# Reweighted ZH samples 
 
samples['ZH_H0PM_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0M_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0M_M0')  
 
samples['ZH_H0PM_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0M_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0M_M1')  
 
samples['ZH_H0PM_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0M_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0M_M2')  
 
samples['ZH_H0PM_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0M_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0M_M3')  
 
samples['ZH_H0PM_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0PH_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0PH_M0')  
 
samples['ZH_H0PM_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0PH_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0PH_M1')  
 
samples['ZH_H0PM_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0PH_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0PH_M2')  
 
samples['ZH_H0PM_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0PH_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0PH_M3')  
 
samples['ZH_H0PM_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0L1_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0L1_M0')  
 
samples['ZH_H0PM_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0L1_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0L1_M1')  
 
samples['ZH_H0PM_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0L1_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0L1_M2')  
 
samples['ZH_H0PM_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0L1_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0L1_M3')  
 
samples['ZH_H0PM_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0LZg_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0LZg_M0')  
 
samples['ZH_H0PM_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0LZg_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0LZg_M1')  
 
samples['ZH_H0PM_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0LZg_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0LZg_M2')  
 
samples['ZH_H0PM_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_H0LZg_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_H0LZg_M3')  
 
samples['ZH_H0PM_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0M_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0M_M0')  
 
samples['ZH_H0PM_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0M_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0M_M1')  
 
samples['ZH_H0PM_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0M_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0M_M2')  
 
samples['ZH_H0PM_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0M_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0M_M3')  
 
samples['ZH_H0PM_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0PH_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0PH_M0')  
 
samples['ZH_H0PM_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0PH_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0PH_M1')  
 
samples['ZH_H0PM_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0PH_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0PH_M2')  
 
samples['ZH_H0PM_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0PH_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0PH_M3')  
 
samples['ZH_H0PM_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0L1_M0/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0L1_M0')  
 
samples['ZH_H0PM_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0L1_M1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0L1_M1')  
 
samples['ZH_H0PM_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0L1_M2/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0L1_M2')  
 
samples['ZH_H0PM_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PM_W*(ME_EFTH0L1_M3/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PM_EFTH0L1_M3')  
 
samples['ZH_H0M_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0PM/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0PM')  
 
samples['ZH_H0M_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0M_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0M_M0')  
 
samples['ZH_H0M_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0M_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0M_M1')  
 
samples['ZH_H0M_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0M_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0M_M2')  
 
samples['ZH_H0M_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0M_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0M_M3')  
 
samples['ZH_H0M_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0PH_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0PH_M0')  
 
samples['ZH_H0M_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0PH_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0PH_M1')  
 
samples['ZH_H0M_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0PH_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0PH_M2')  
 
samples['ZH_H0M_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0PH_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0PH_M3')  
 
samples['ZH_H0M_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0L1_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0L1_M0')  
 
samples['ZH_H0M_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0L1_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0L1_M1')  
 
samples['ZH_H0M_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0L1_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0L1_M2')  
 
samples['ZH_H0M_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0L1_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0L1_M3')  
 
samples['ZH_H0M_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0LZg_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0LZg_M0')  
 
samples['ZH_H0M_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0LZg_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0LZg_M1')  
 
samples['ZH_H0M_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0LZg_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0LZg_M2')  
 
samples['ZH_H0M_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_H0LZg_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_H0LZg_M3')  
 
samples['ZH_H0M_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0M_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0M_M0')  
 
samples['ZH_H0M_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0M_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0M_M1')  
 
samples['ZH_H0M_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0M_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0M_M2')  
 
samples['ZH_H0M_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0M_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0M_M3')  
 
samples['ZH_H0M_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0PH_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0PH_M0')  
 
samples['ZH_H0M_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0PH_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0PH_M1')  
 
samples['ZH_H0M_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0PH_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0PH_M2')  
 
samples['ZH_H0M_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0PH_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0PH_M3')  
 
samples['ZH_H0M_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0L1_M0/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0L1_M0')  
 
samples['ZH_H0M_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0L1_M1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0L1_M1')  
 
samples['ZH_H0M_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0L1_M2/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0L1_M2')  
 
samples['ZH_H0M_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0M_W*(ME_EFTH0L1_M3/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0M_EFTH0L1_M3')  
 
samples['ZH_H0Mf05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0PM/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0PM')  
 
samples['ZH_H0Mf05_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0M_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0M_M0')  
 
samples['ZH_H0Mf05_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0M_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0M_M1')  
 
samples['ZH_H0Mf05_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0M_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0M_M2')  
 
samples['ZH_H0Mf05_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0M_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0M_M3')  
 
samples['ZH_H0Mf05_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0PH_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0PH_M0')  
 
samples['ZH_H0Mf05_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0PH_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0PH_M1')  
 
samples['ZH_H0Mf05_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0PH_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0PH_M2')  
 
samples['ZH_H0Mf05_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0PH_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0PH_M3')  
 
samples['ZH_H0Mf05_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0L1_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0L1_M0')  
 
samples['ZH_H0Mf05_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0L1_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0L1_M1')  
 
samples['ZH_H0Mf05_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0L1_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0L1_M2')  
 
samples['ZH_H0Mf05_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0L1_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0L1_M3')  
 
samples['ZH_H0Mf05_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0LZg_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0LZg_M0')  
 
samples['ZH_H0Mf05_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0LZg_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0LZg_M1')  
 
samples['ZH_H0Mf05_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0LZg_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0LZg_M2')  
 
samples['ZH_H0Mf05_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_H0LZg_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_H0LZg_M3')  
 
samples['ZH_H0Mf05_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0M_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0M_M0')  
 
samples['ZH_H0Mf05_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0M_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0M_M1')  
 
samples['ZH_H0Mf05_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0M_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0M_M2')  
 
samples['ZH_H0Mf05_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0M_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0M_M3')  
 
samples['ZH_H0Mf05_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0PH_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0PH_M0')  
 
samples['ZH_H0Mf05_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0PH_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0PH_M1')  
 
samples['ZH_H0Mf05_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0PH_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0PH_M2')  
 
samples['ZH_H0Mf05_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0PH_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0PH_M3')  
 
samples['ZH_H0Mf05_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0L1_M0/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0L1_M0')  
 
samples['ZH_H0Mf05_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0L1_M1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0L1_M1')  
 
samples['ZH_H0Mf05_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0L1_M2/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0L1_M2')  
 
samples['ZH_H0Mf05_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0Mf05_W*(ME_EFTH0L1_M3/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0Mf05_EFTH0L1_M3')  
 
samples['ZH_H0PH_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0PM/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0PM')  
 
samples['ZH_H0PH_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0M_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0M_M0')  
 
samples['ZH_H0PH_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0M_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0M_M1')  
 
samples['ZH_H0PH_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0M_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0M_M2')  
 
samples['ZH_H0PH_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0M_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0M_M3')  
 
samples['ZH_H0PH_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0PH_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0PH_M0')  
 
samples['ZH_H0PH_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0PH_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0PH_M1')  
 
samples['ZH_H0PH_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0PH_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0PH_M2')  
 
samples['ZH_H0PH_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0PH_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0PH_M3')  
 
samples['ZH_H0PH_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0L1_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0L1_M0')  
 
samples['ZH_H0PH_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0L1_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0L1_M1')  
 
samples['ZH_H0PH_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0L1_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0L1_M2')  
 
samples['ZH_H0PH_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0L1_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0L1_M3')  
 
samples['ZH_H0PH_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0LZg_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0LZg_M0')  
 
samples['ZH_H0PH_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0LZg_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0LZg_M1')  
 
samples['ZH_H0PH_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0LZg_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0LZg_M2')  
 
samples['ZH_H0PH_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_H0LZg_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_H0LZg_M3')  
 
samples['ZH_H0PH_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0M_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0M_M0')  
 
samples['ZH_H0PH_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0M_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0M_M1')  
 
samples['ZH_H0PH_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0M_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0M_M2')  
 
samples['ZH_H0PH_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0M_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0M_M3')  
 
samples['ZH_H0PH_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0PH_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0PH_M0')  
 
samples['ZH_H0PH_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0PH_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0PH_M1')  
 
samples['ZH_H0PH_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0PH_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0PH_M2')  
 
samples['ZH_H0PH_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0PH_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0PH_M3')  
 
samples['ZH_H0PH_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0L1_M0/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0L1_M0')  
 
samples['ZH_H0PH_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0L1_M1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0L1_M1')  
 
samples['ZH_H0PH_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0L1_M2/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0L1_M2')  
 
samples['ZH_H0PH_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PH_W*(ME_EFTH0L1_M3/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PH_EFTH0L1_M3')  
 
samples['ZH_H0PHf05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0PM/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0PM')  
 
samples['ZH_H0PHf05_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0M_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0M_M0')  
 
samples['ZH_H0PHf05_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0M_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0M_M1')  
 
samples['ZH_H0PHf05_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0M_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0M_M2')  
 
samples['ZH_H0PHf05_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0M_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0M_M3')  
 
samples['ZH_H0PHf05_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0PH_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0PH_M0')  
 
samples['ZH_H0PHf05_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0PH_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0PH_M1')  
 
samples['ZH_H0PHf05_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0PH_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0PH_M2')  
 
samples['ZH_H0PHf05_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0PH_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0PH_M3')  
 
samples['ZH_H0PHf05_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0L1_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0L1_M0')  
 
samples['ZH_H0PHf05_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0L1_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0L1_M1')  
 
samples['ZH_H0PHf05_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0L1_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0L1_M2')  
 
samples['ZH_H0PHf05_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0L1_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0L1_M3')  
 
samples['ZH_H0PHf05_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0LZg_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0LZg_M0')  
 
samples['ZH_H0PHf05_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0LZg_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0LZg_M1')  
 
samples['ZH_H0PHf05_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0LZg_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0LZg_M2')  
 
samples['ZH_H0PHf05_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_H0LZg_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_H0LZg_M3')  
 
samples['ZH_H0PHf05_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0M_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0M_M0')  
 
samples['ZH_H0PHf05_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0M_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0M_M1')  
 
samples['ZH_H0PHf05_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0M_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0M_M2')  
 
samples['ZH_H0PHf05_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0M_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0M_M3')  
 
samples['ZH_H0PHf05_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0PH_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0PH_M0')  
 
samples['ZH_H0PHf05_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0PH_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0PH_M1')  
 
samples['ZH_H0PHf05_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0PH_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0PH_M2')  
 
samples['ZH_H0PHf05_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0PH_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0PH_M3')  
 
samples['ZH_H0PHf05_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0L1_M0/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0L1_M0')  
 
samples['ZH_H0PHf05_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0L1_M1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0L1_M1')  
 
samples['ZH_H0PHf05_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0L1_M2/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0L1_M2')  
 
samples['ZH_H0PHf05_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0PHf05_W*(ME_EFTH0L1_M3/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0PHf05_EFTH0L1_M3')  
 
samples['ZH_H0L1_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0PM/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0PM')  
 
samples['ZH_H0L1_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0M_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0M_M0')  
 
samples['ZH_H0L1_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0M_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0M_M1')  
 
samples['ZH_H0L1_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0M_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0M_M2')  
 
samples['ZH_H0L1_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0M_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0M_M3')  
 
samples['ZH_H0L1_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0PH_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0PH_M0')  
 
samples['ZH_H0L1_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0PH_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0PH_M1')  
 
samples['ZH_H0L1_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0PH_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0PH_M2')  
 
samples['ZH_H0L1_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0PH_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0PH_M3')  
 
samples['ZH_H0L1_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0L1_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0L1_M0')  
 
samples['ZH_H0L1_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0L1_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0L1_M1')  
 
samples['ZH_H0L1_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0L1_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0L1_M2')  
 
samples['ZH_H0L1_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0L1_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0L1_M3')  
 
samples['ZH_H0L1_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0LZg_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0LZg_M0')  
 
samples['ZH_H0L1_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0LZg_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0LZg_M1')  
 
samples['ZH_H0L1_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0LZg_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0LZg_M2')  
 
samples['ZH_H0L1_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_H0LZg_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_H0LZg_M3')  
 
samples['ZH_H0L1_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0M_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0M_M0')  
 
samples['ZH_H0L1_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0M_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0M_M1')  
 
samples['ZH_H0L1_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0M_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0M_M2')  
 
samples['ZH_H0L1_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0M_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0M_M3')  
 
samples['ZH_H0L1_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0PH_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0PH_M0')  
 
samples['ZH_H0L1_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0PH_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0PH_M1')  
 
samples['ZH_H0L1_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0PH_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0PH_M2')  
 
samples['ZH_H0L1_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0PH_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0PH_M3')  
 
samples['ZH_H0L1_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0L1_M0/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0L1_M0')  
 
samples['ZH_H0L1_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0L1_M1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0L1_M1')  
 
samples['ZH_H0L1_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0L1_M2/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0L1_M2')  
 
samples['ZH_H0L1_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1_W*(ME_EFTH0L1_M3/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1_EFTH0L1_M3')  
 
samples['ZH_H0L1f05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0PM/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0PM')  
 
samples['ZH_H0L1f05_H0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0M_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0M_M0')  
 
samples['ZH_H0L1f05_H0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0M_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0M_M1')  
 
samples['ZH_H0L1f05_H0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0M_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0M_M2')  
 
samples['ZH_H0L1f05_H0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0M_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0M_M3')  
 
samples['ZH_H0L1f05_H0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0PH_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0PH_M0')  
 
samples['ZH_H0L1f05_H0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0PH_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0PH_M1')  
 
samples['ZH_H0L1f05_H0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0PH_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0PH_M2')  
 
samples['ZH_H0L1f05_H0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0PH_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0PH_M3')  
 
samples['ZH_H0L1f05_H0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0L1_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0L1_M0')  
 
samples['ZH_H0L1f05_H0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0L1_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0L1_M1')  
 
samples['ZH_H0L1f05_H0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0L1_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0L1_M2')  
 
samples['ZH_H0L1f05_H0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0L1_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0L1_M3')  
 
samples['ZH_H0L1f05_H0LZg_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0LZg_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0LZg_M0')  
 
samples['ZH_H0L1f05_H0LZg_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0LZg_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0LZg_M1')  
 
samples['ZH_H0L1f05_H0LZg_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0LZg_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0LZg_M2')  
 
samples['ZH_H0L1f05_H0LZg_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_H0LZg_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_H0LZg_M3')  
 
samples['ZH_H0L1f05_EFTH0M_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0M_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0M_M0')  
 
samples['ZH_H0L1f05_EFTH0M_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0M_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0M_M1')  
 
samples['ZH_H0L1f05_EFTH0M_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0M_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0M_M2')  
 
samples['ZH_H0L1f05_EFTH0M_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0M_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0M_M3')  
 
samples['ZH_H0L1f05_EFTH0PH_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0PH_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0PH_M0')  
 
samples['ZH_H0L1f05_EFTH0PH_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0PH_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0PH_M1')  
 
samples['ZH_H0L1f05_EFTH0PH_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0PH_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0PH_M2')  
 
samples['ZH_H0L1f05_EFTH0PH_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0PH_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0PH_M3')  
 
samples['ZH_H0L1f05_EFTH0L1_M0'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0L1_M0/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0L1_M0')  
 
samples['ZH_H0L1f05_EFTH0L1_M1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0L1_M1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0L1_M1')  
 
samples['ZH_H0L1f05_EFTH0L1_M2'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0L1_M2/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0L1_M2')  
 
samples['ZH_H0L1f05_EFTH0L1_M3'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'ZH_H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*ZH_H0L1f05_W*(ME_EFTH0L1_M3/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('ZH_H0L1f05_EFTH0L1_M3')  
 
# GGH MC samples 
 
# Original GGH samples 
 
samples['H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W',   'FilesPerJob': 4, } 
 
samples['H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W',   'FilesPerJob': 4, } 
 
samples['H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W',   'FilesPerJob': 4, } 
 
samples['H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W',   'FilesPerJob': 4, } 
 
samples['H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W',   'FilesPerJob': 4, } 
 
samples['H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W',   'FilesPerJob': 4, } 
 
samples['H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W',   'FilesPerJob': 4, } 
 
# Reweighted GGH samples 
 
samples['H0PM_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_H0M/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_H0M')  
 
samples['H0PM_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_H0Mf05/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_H0Mf05')  
 
samples['H0PM_H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_H0PH/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_H0PH')  
 
samples['H0PM_H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_H0PHf05/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_H0PHf05')  
 
samples['H0PM_H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_H0L1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_H0L1')  
 
samples['H0PM_H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_H0L1f05/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_H0L1f05')  
 
samples['H0PM_EFTH0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_EFTH0M/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_EFTH0M')  
 
samples['H0PM_EFTH0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_EFTH0Mf05/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_EFTH0Mf05')  
 
samples['H0PM_EFTH0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_EFTH0PH/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_EFTH0PH')  
 
samples['H0PM_EFTH0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_EFTH0PHf05/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_EFTH0PHf05')  
 
samples['H0PM_EFTH0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_EFTH0L1/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_EFTH0L1')  
 
samples['H0PM_EFTH0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PM_W*(ME_EFTH0L1f05/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('H0PM_EFTH0L1f05')  
 
samples['H0M_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_H0PM/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_H0PM')  
 
samples['H0M_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_H0Mf05/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_H0Mf05')  
 
samples['H0M_H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_H0PH/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_H0PH')  
 
samples['H0M_H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_H0PHf05/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_H0PHf05')  
 
samples['H0M_H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_H0L1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_H0L1')  
 
samples['H0M_H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_H0L1f05/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_H0L1f05')  
 
samples['H0M_EFTH0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_EFTH0M/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_EFTH0M')  
 
samples['H0M_EFTH0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_EFTH0Mf05/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_EFTH0Mf05')  
 
samples['H0M_EFTH0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_EFTH0PH/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_EFTH0PH')  
 
samples['H0M_EFTH0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_EFTH0PHf05/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_EFTH0PHf05')  
 
samples['H0M_EFTH0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_EFTH0L1/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_EFTH0L1')  
 
samples['H0M_EFTH0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0M_W*(ME_EFTH0L1f05/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('H0M_EFTH0L1f05')  
 
samples['H0Mf05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_H0PM/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_H0PM')  
 
samples['H0Mf05_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_H0M/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_H0M')  
 
samples['H0Mf05_H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_H0PH/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_H0PH')  
 
samples['H0Mf05_H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_H0PHf05/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_H0PHf05')  
 
samples['H0Mf05_H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_H0L1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_H0L1')  
 
samples['H0Mf05_H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_H0L1f05/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_H0L1f05')  
 
samples['H0Mf05_EFTH0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_EFTH0M/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_EFTH0M')  
 
samples['H0Mf05_EFTH0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_EFTH0Mf05/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_EFTH0Mf05')  
 
samples['H0Mf05_EFTH0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_EFTH0PH/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_EFTH0PH')  
 
samples['H0Mf05_EFTH0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_EFTH0PHf05/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_EFTH0PHf05')  
 
samples['H0Mf05_EFTH0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_EFTH0L1/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_EFTH0L1')  
 
samples['H0Mf05_EFTH0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0Mf05_W*(ME_EFTH0L1f05/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0Mf05_EFTH0L1f05')  
 
samples['H0PH_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_H0PM/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_H0PM')  
 
samples['H0PH_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_H0M/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_H0M')  
 
samples['H0PH_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_H0Mf05/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_H0Mf05')  
 
samples['H0PH_H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_H0PHf05/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_H0PHf05')  
 
samples['H0PH_H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_H0L1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_H0L1')  
 
samples['H0PH_H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_H0L1f05/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_H0L1f05')  
 
samples['H0PH_EFTH0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_EFTH0M/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_EFTH0M')  
 
samples['H0PH_EFTH0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_EFTH0Mf05/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_EFTH0Mf05')  
 
samples['H0PH_EFTH0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_EFTH0PH/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_EFTH0PH')  
 
samples['H0PH_EFTH0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_EFTH0PHf05/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_EFTH0PHf05')  
 
samples['H0PH_EFTH0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_EFTH0L1/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_EFTH0L1')  
 
samples['H0PH_EFTH0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PH_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PH_W*(ME_EFTH0L1f05/ME_H0PH)',   'FilesPerJob': 4, } 
signals_rw.append('H0PH_EFTH0L1f05')  
 
samples['H0PHf05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_H0PM/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_H0PM')  
 
samples['H0PHf05_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_H0M/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_H0M')  
 
samples['H0PHf05_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_H0Mf05/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_H0Mf05')  
 
samples['H0PHf05_H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_H0PH/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_H0PH')  
 
samples['H0PHf05_H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_H0L1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_H0L1')  
 
samples['H0PHf05_H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_H0L1f05/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_H0L1f05')  
 
samples['H0PHf05_EFTH0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_EFTH0M/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_EFTH0M')  
 
samples['H0PHf05_EFTH0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_EFTH0Mf05/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_EFTH0Mf05')  
 
samples['H0PHf05_EFTH0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_EFTH0PH/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_EFTH0PH')  
 
samples['H0PHf05_EFTH0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_EFTH0PHf05/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_EFTH0PHf05')  
 
samples['H0PHf05_EFTH0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_EFTH0L1/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_EFTH0L1')  
 
samples['H0PHf05_EFTH0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0PHf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0PHf05_W*(ME_EFTH0L1f05/ME_H0PHf05)',   'FilesPerJob': 4, } 
signals_rw.append('H0PHf05_EFTH0L1f05')  
 
samples['H0L1_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_H0PM/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_H0PM')  
 
samples['H0L1_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_H0M/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_H0M')  
 
samples['H0L1_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_H0Mf05/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_H0Mf05')  
 
samples['H0L1_H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_H0PH/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_H0PH')  
 
samples['H0L1_H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_H0PHf05/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_H0PHf05')  
 
samples['H0L1_H0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_H0L1f05/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_H0L1f05')  
 
samples['H0L1_EFTH0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_EFTH0M/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_EFTH0M')  
 
samples['H0L1_EFTH0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_EFTH0Mf05/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_EFTH0Mf05')  
 
samples['H0L1_EFTH0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_EFTH0PH/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_EFTH0PH')  
 
samples['H0L1_EFTH0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_EFTH0PHf05/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_EFTH0PHf05')  
 
samples['H0L1_EFTH0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_EFTH0L1/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_EFTH0L1')  
 
samples['H0L1_EFTH0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1_W*(ME_EFTH0L1f05/ME_H0L1)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1_EFTH0L1f05')  
 
samples['H0L1f05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_H0PM/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_H0PM')  
 
samples['H0L1f05_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_H0M/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_H0M')  
 
samples['H0L1f05_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_H0Mf05/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_H0Mf05')  
 
samples['H0L1f05_H0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_H0PH/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_H0PH')  
 
samples['H0L1f05_H0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_H0PHf05/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_H0PHf05')  
 
samples['H0L1f05_H0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_H0L1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_H0L1')  
 
samples['H0L1f05_EFTH0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_EFTH0M/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_EFTH0M')  
 
samples['H0L1f05_EFTH0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_EFTH0Mf05/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_EFTH0Mf05')  
 
samples['H0L1f05_EFTH0PH'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_EFTH0PH/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_EFTH0PH')  
 
samples['H0L1f05_EFTH0PHf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_EFTH0PHf05/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_EFTH0PHf05')  
 
samples['H0L1f05_EFTH0L1'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_EFTH0L1/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_EFTH0L1')  
 
samples['H0L1f05_EFTH0L1f05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'H0L1f05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*H0L1f05_W*(ME_EFTH0L1f05/ME_H0L1f05)',   'FilesPerJob': 4, } 
signals_rw.append('H0L1f05_EFTH0L1f05')  
 
# GGHjj MC samples 
 
# Original GGHjj samples 
 
samples['GGHjj_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'GGHjj_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*GGHjj_H0PM_W',   'FilesPerJob': 4, } 
 
samples['GGHjj_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'GGHjj_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*GGHjj_H0M_W',   'FilesPerJob': 4, } 
 
samples['GGHjj_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'GGHjj_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*GGHjj_H0Mf05_W',   'FilesPerJob': 4, } 
 
# Reweighted GGHjj samples 
 
samples['GGHjj_H0PM_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'GGHjj_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*GGHjj_H0PM_W*(ME_H0M/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('GGHjj_H0PM_H0M')  
 
samples['GGHjj_H0PM_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'GGHjj_H0PM_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*GGHjj_H0PM_W*(ME_H0Mf05/ME_H0PM)',   'FilesPerJob': 4, } 
signals_rw.append('GGHjj_H0PM_H0Mf05')  
 
samples['GGHjj_H0M_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'GGHjj_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*GGHjj_H0M_W*(ME_H0PM/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('GGHjj_H0M_H0PM')  
 
samples['GGHjj_H0M_H0Mf05'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'GGHjj_H0M_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*GGHjj_H0M_W*(ME_H0Mf05/ME_H0M)',   'FilesPerJob': 4, } 
signals_rw.append('GGHjj_H0M_H0Mf05')  
 
samples['GGHjj_H0Mf05_H0PM'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'GGHjj_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*GGHjj_H0Mf05_W*(ME_H0PM/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('GGHjj_H0Mf05_H0PM')  
 
samples['GGHjj_H0Mf05_H0M'] = { 
   'name':   nanoGetSampleFiles(mcDirectory, 'GGHjj_H0Mf05_ToWWTo2L2Nu'), 
   'weight': mcCommonWeight+ '*GGHjj_H0Mf05_W*(ME_H0M/ME_H0Mf05)',   'FilesPerJob': 4, } 
signals_rw.append('GGHjj_H0Mf05_H0M')  
 
