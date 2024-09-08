import os
import subprocess

global getSampleFiles
from LatinoAnalysis.Tools.commonTools import getSampleFiles, addSampleWeight, getBaseWnAOD

def getSampleFilesNano(inputDir,Sample,absPath=False):
    return getSampleFiles(inputDir,Sample,absPath,'nanoLatino_')

##############################################
###### Tree Directory according to site ######
##############################################

SITE=os.uname()[1]
xrootdPath=''
if    'iihe' in SITE :
  xrootdPath  = 'dcap://maite.iihe.ac.be/'
  treeBaseDir = '/pnfs/iihe/cms/store/user/xjanssen/HWW2015/'
elif  'cern' in SITE :
  #xrootdPath='root://eoscms.cern.ch/'
  treeBaseDir = '/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano/'

directory = treeBaseDir+'Fall2017_102X_nAODv7_Full2017v7/MCl1loose2017v7__MCCorr2017v7__l2loose__l2tightOR2017v7'

################################################
############ NUMBER OF LEPTONS #################
################################################

#Nlep='2'
Nlep='3'
#Nlep='4'

################################################
############### Lepton WP ######################
################################################

eleWP='mvaFall17V1Iso_WP90'
#eleWP='mvaFall17V1Iso_WP90_SS'
#eleWP='mvaFall17V2Iso_WP90'
#eleWP='mvaFall17V2Iso_WP90_SS'
muWP ='cut_Tight_HWWW'
eleWP_new = 'mvaFall17V1Iso_WP90_tthmva_70'
muWP_new  = 'cut_Tight_HWWW_tthmva_80'

#LepWPCut        = 'LepCut'+Nlep+'l__ele_'+eleWP+'__mu_'+muWP+'*LepWPCutNew' #Cut for new WPs, defined in aliases 
LepWPCut         = 'LepCut'+Nlep+'l__ele_'+eleWP_new+'__mu_'+muWP_new  #Cut for new WPs, v7
#LepWPCut        = 'LepCut'+Nlep+'l__ele_'+eleWP+'__mu_'+muWP
#LepWPweight     = 'ttHMVA_SF_3l[0]' #SF for new WPs, defined in aliases
LepWPweight      = 'LepSF'+Nlep+'l__ele_'+eleWP_new+'__mu_'+muWP_new
#LepWPweight     = 'LepSF'+Nlep+'l__ele_'+eleWP+'__mu_'+muWP

################################################
############ BASIC MC WEIGHTS ##################
################################################

XSWeight      = 'XSWeight'
#SFweight      = 'SFweight'+Nlep+'l*'+LepWPweight+'*'+LepWPCut+'*PrefireWeight*PUJetIdSF'
SFweight      = 'SFweight'+Nlep+'l*'+LepWPweight+'*'+LepWPCut+'*PrefireWeight*Jet_PUIDSF'
PromptGenLepMatch   = 'PromptGenLepMatch'+Nlep+'l'

################################################
############## FAKE WEIGHTS ####################
################################################

#eleWP_new = 'mvaFall17V1Iso_WP90_tthmva_70'
#muWP_new  = 'cut_Tight_HWWW_tthmva_80'

if Nlep == '2' :
  fakeW = 'fakeW2l_ele_'+eleWP_new+'_mu_'+muWP_new
  #fakeW = 'fakeW2l_ele_'+eleWP+'_mu_'+muWP
else:
  fakeW = 'fakeW_ele_'+eleWP_new+'_mu_'+muWP_new+'_'+Nlep+'l'
  #fakeW = 'fakeW_ele_'+eleWP+'_mu_'+muWP+'_'+Nlep+'l'

################################################
############### B-Tag  WP ######################
################################################

SFweight += '*btagSF' #define in aliases.py

################################################
############   MET  FILTERS  ###################
################################################

METFilter_MC   = 'METFilter_MC'
METFilter_DATA = 'METFilter_DATA'

################################################
############ DATA DECLARATION ##################
################################################

DataRun = [
    ['B','Run2017B-02Apr2020-v1'],
    ['C','Run2017C-02Apr2020-v1'],
    ['D','Run2017D-02Apr2020-v1'],
    ['E','Run2017E-02Apr2020-v1'],
    ['F','Run2017F-02Apr2020-v1']
]

DataSets = ['MuonEG','DoubleMuon','SingleMuon','DoubleEG','SingleElectron']

DataTrig = {
            'MuonEG'         : ' Trigger_ElMu' ,
            'DoubleMuon'     : '!Trigger_ElMu &&  Trigger_dblMu' ,
            'SingleMuon'     : '!Trigger_ElMu && !Trigger_dblMu &&  Trigger_sngMu' ,
            'DoubleEG'       : '!Trigger_ElMu && !Trigger_dblMu && !Trigger_sngMu &&  Trigger_dblEl' ,
            'SingleElectron' : '!Trigger_ElMu && !Trigger_dblMu && !Trigger_sngMu && !Trigger_dblEl && Trigger_sngEl' ,
           }

###########################################
#############  BACKGROUNDS  ###############
###########################################
ptllDYW_NLO = '(((0.623108 + 0.0722934*gen_ptll - 0.00364918*gen_ptll*gen_ptll + 6.97227e-05*gen_ptll*gen_ptll*gen_ptll - 4.52903e-07*gen_ptll*gen_ptll*gen_ptll*gen_ptll)*(gen_ptll<45)*(gen_ptll>0) + 1*(gen_ptll>=45))*(abs(gen_mll-90)<3) + (abs(gen_mll-90)>3))'
ptllDYW_LO = '((0.632927+0.0456956*gen_ptll-0.00154485*gen_ptll*gen_ptll+2.64397e-05*gen_ptll*gen_ptll*gen_ptll-2.19374e-07*gen_ptll*gen_ptll*gen_ptll*gen_ptll+6.99751e-10*gen_ptll*gen_ptll*gen_ptll*gen_ptll*gen_ptll)*(gen_ptll>0)*(gen_ptll<100)+(1.41713-0.00165342*gen_ptll)*(gen_ptll>=100)*(gen_ptll<300)+1*(gen_ptll>=300))'
Zgfilter    = '( !(Sum$(PhotonGen_isPrompt==1 && PhotonGen_pt>15 && abs(PhotonGen_eta)<2.6) > 0 && Sum$(LeptonGen_isPrompt==1 && LeptonGen_pt>15)>=2) )' #Zg sample uses photon pt > 15, lepton pt > 15

samples['DY'] = {    'name'   :    getSampleFilesNano(directory,'DYJetsToLL_M-10to50-LO_ext1')
                                #+ getSampleFilesNano(directory,'DYJetsToLL_M-10to50-LO') #missing for v7
                                 + getSampleFilesNano(directory,'DYJetsToLL_M-50-LO')
                                 + getSampleFilesNano(directory,'DYJetsToLL_M-50-LO_ext1')
                                #+ getSampleFilesNano(directory,'DYJetsToLL_M-4to50_HT-100to200') #missing for v7
                                 + getSampleFilesNano(directory,'DYJetsToLL_M-4to50_HT-100to200_ext1')
                                #+ getSampleFilesNano(directory,'DYJetsToLL_M-4to50_HT-200to400')
                                #+ getSampleFilesNano(directory,'DYJetsToLL_M-4to50_HT-200to400_ext1')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-4to50_HT-400to600')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-4to50_HT-400to600_ext1')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-4to50_HT-600toInf')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-4to50_HT-600toInf_ext1')
                               #+ getSampleFilesNano(directory,'DYJetsToLL_M-50_HT-70to100') #Drop for now -- missing files for METup variation
                                + getSampleFilesNano(directory,'DYJetsToLL_M-50_HT-100to200')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-50_HT-200to400')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-50_HT-200to400_ext1')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-50_HT-400to600_ext1')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-50_HT-600to800')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-50_HT-800to1200')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-50_HT-1200to2500')
                                + getSampleFilesNano(directory,'DYJetsToLL_M-50_HT-2500toInf'),
                       'weight' : XSWeight+'*'+SFweight+'*'+PromptGenLepMatch+'*'+METFilter_MC+'*'+Zgfilter ,
                       'FilesPerJob' : 5,
                       'suppressNegative' :['all'],
                       'suppressNegativeNuisances' :['all'],
															                  }

#M10baseW      = getBaseWnAOD(directory,'Fall2017_102X_nAODv5_Full2017v6',['DYJetsToLL_M-10to50-LO',        'DYJetsToLL_M-10to50-LO_ext1'])
M10baseW      = getBaseWnAOD(directory,'Fall2017_102X_nAODv7_Full2017v7',['DYJetsToLL_M-10to50-LO_ext1'])
M50baseW      = getBaseWnAOD(directory,'Fall2017_102X_nAODv7_Full2017v7',['DYJetsToLL_M-50-LO',            'DYJetsToLL_M-50-LO_ext1'])
#HT100M4baseW  = getBaseWnAOD(directory,'Fall2017_102X_nAODv5_Full2017v6',['DYJetsToLL_M-4to50_HT-100to200','DYJetsToLL_M-4to50_HT-100to200_ext1'])
HT100M4baseW  = getBaseWnAOD(directory,'Fall2017_102X_nAODv7_Full2017v7',['DYJetsToLL_M-4to50_HT-100to200_ext1'])
#HT200M4baseW  = getBaseWnAOD(directory,'Fall2017_102X_nAODv5_Full2017v6',['DYJetsToLL_M-4to50_HT-200to400','DYJetsToLL_M-4to50_HT-200to400_ext1'])
HT400M4baseW  = getBaseWnAOD(directory,'Fall2017_102X_nAODv7_Full2017v7',['DYJetsToLL_M-4to50_HT-400to600','DYJetsToLL_M-4to50_HT-400to600_ext1'])
HT600M4baseW  = getBaseWnAOD(directory,'Fall2017_102X_nAODv7_Full2017v7',['DYJetsToLL_M-4to50_HT-600toInf','DYJetsToLL_M-4to50_HT-600toInf_ext1'])
HT200M50baseW = getBaseWnAOD(directory,'Fall2017_102X_nAODv7_Full2017v7',['DYJetsToLL_M-50_HT-200to400',   'DYJetsToLL_M-50_HT-200to400_ext1'])

#addSampleWeight(samples,'DY','DYJetsToLL_M-10to50-LO'              ,ptllDYW_LO+'*(LHE_HT<100.0)*'+M10baseW+'/baseW')
addSampleWeight(samples,'DY','DYJetsToLL_M-10to50-LO_ext1'         ,ptllDYW_LO+'*(LHE_HT<100.0)*'+M10baseW+'/baseW')
addSampleWeight(samples,'DY','DYJetsToLL_M-50-LO'                  ,ptllDYW_LO+'*(LHE_HT<100.0)*'+M50baseW+'/baseW') #TODO change when adding back HT70to100
addSampleWeight(samples,'DY','DYJetsToLL_M-50-LO_ext1'             ,ptllDYW_LO+'*(LHE_HT<100.0)*'+M50baseW+'/baseW') #TODO
#addSampleWeight(samples,'DY','DYJetsToLL_M-4to50_HT-100to200'      ,ptllDYW_LO+'*'+HT100M4baseW+'/baseW')
addSampleWeight(samples,'DY','DYJetsToLL_M-4to50_HT-100to200_ext1' ,ptllDYW_LO+'*'+HT100M4baseW+'/baseW')
#addSampleWeight(samples,'DY','DYJetsToLL_M-4to50_HT-200to400'      ,ptllDYW_LO+'*'+HT200M4baseW+'/baseW')
#addSampleWeight(samples,'DY','DYJetsToLL_M-4to50_HT-200to400_ext1' ,ptllDYW_LO+'*'+HT200M4baseW+'/baseW')
addSampleWeight(samples,'DY','DYJetsToLL_M-4to50_HT-400to600'      ,ptllDYW_LO+'*'+HT400M4baseW+'/baseW')
addSampleWeight(samples,'DY','DYJetsToLL_M-4to50_HT-400to600_ext1' ,ptllDYW_LO+'*'+HT400M4baseW+'/baseW')
addSampleWeight(samples,'DY','DYJetsToLL_M-4to50_HT-600toInf'      ,ptllDYW_LO+'*'+HT600M4baseW+'/baseW')
addSampleWeight(samples,'DY','DYJetsToLL_M-4to50_HT-600toInf_ext1' ,ptllDYW_LO+'*'+HT600M4baseW+'/baseW')
#addSampleWeight(samples,'DY','DYJetsToLL_M-50_HT-70to100'          ,ptllDYW_LO) #TODO
addSampleWeight(samples,'DY','DYJetsToLL_M-50_HT-100to200'         ,ptllDYW_LO)
addSampleWeight(samples,'DY','DYJetsToLL_M-50_HT-200to400'         ,ptllDYW_LO+'*'+HT200M50baseW+'/baseW')
addSampleWeight(samples,'DY','DYJetsToLL_M-50_HT-200to400_ext1'    ,ptllDYW_LO+'*'+HT200M50baseW+'/baseW')
addSampleWeight(samples,'DY','DYJetsToLL_M-50_HT-400to600_ext1'    ,ptllDYW_LO)
addSampleWeight(samples,'DY','DYJetsToLL_M-50_HT-600to800'         ,ptllDYW_LO)
addSampleWeight(samples,'DY','DYJetsToLL_M-50_HT-800to1200'        ,ptllDYW_LO)
addSampleWeight(samples,'DY','DYJetsToLL_M-50_HT-1200to2500'       ,ptllDYW_LO)
addSampleWeight(samples,'DY','DYJetsToLL_M-50_HT-2500toInf'        ,ptllDYW_LO)




samples['Zg']  =  {     'name'   :    getSampleFilesNano(directory,'ZGToLLG'),
                        'weight' : XSWeight+'*'+SFweight+'*'+METFilter_MC + '*(Gen_ZGstar_mass <= 0)',
                        'FilesPerJob' : 3 ,
                  }

samples['ZgS']  = {    'name'   :   getSampleFilesNano(directory,'ZGToLLG'),
                       'weight' : XSWeight+'*'+SFweight+'*'+PromptGenLepMatch+'*'+METFilter_MC+'*(Gen_ZGstar_mass > 0)',
                       'FilesPerJob' : 3 ,
                  }




samples['WZ']  = {    'name':   getSampleFilesNano(directory,'WZTo3LNu_mllmin01')
                              + getSampleFilesNano(directory,'WZTo2L2Q'),
                       'weight' : XSWeight+'*'+SFweight+'*'+PromptGenLepMatch+'*'+METFilter_MC+'*(gstarHigh)' ,
                       'FilesPerJob' : 2 ,
             }


samples['ttV'] = {    'name'   :   getSampleFilesNano(directory,'TTWJetsToLNu_PSweights') #missing TTWJetsToLNu for v7
                                 + getSampleFilesNano(directory,'TTZjets')
                                 + getSampleFilesNano(directory,'TTZjets_ext1'),
                     'weight' : XSWeight+'*'+SFweight+'*'+PromptGenLepMatch+'*'+METFilter_MC ,
                     'FilesPerJob' : 5,
                 }

ttZbaseW = getBaseWnAOD(directory,'Fall2017_102X_nAODv7_Full2017v7',['TTZjets','TTZjets_ext1'])

addSampleWeight(samples,'ttV','TTZjets'     ,ttZbaseW+'/baseW*1.0989')
addSampleWeight(samples,'ttV','TTZjets_ext1',ttZbaseW+'/baseW*1.0989')

############ VVV ############

samples['ZZ']  = {  'name'   :   getSampleFilesNano(directory,'ZZTo2L2Nu')
                               + getSampleFilesNano(directory,'ZZTo2L2Q')
                               + getSampleFilesNano(directory,'ZZTo4L')
                              #+ getSampleFilesNano(directory,'ZZTo4L_ext1') #Missing for v7
                              #+ getSampleFilesNano(directory,'ZZTo4L_ext2') #Missing for v7
                              #+ getSampleFilesNano(directory,'ggZZ4m')      #Corrupt file for ElepTup
                              #+ getSampleFilesNano(directory,'ggZZ4m_ext1') #Missing file for ElepTup
                               + getSampleFilesNano(directory,'ggZZ4m_ext1') #Missing ext2 for v7
                               + getSampleFilesNano(directory,'ggZZ2e2t_ext1')
                              #+ getSampleFilesNano(directory,'ggZZ2m2t') #Missing for v7
                               + getSampleFilesNano(directory,'ggZZ2m2t_ext1')
                              #+ getSampleFilesNano(directory,'ggZZ2e2m')    #Corrupt file for ElepTup
                               + getSampleFilesNano(directory,'ggZZ2e2m_ext1'),
                    'weight' : XSWeight+'*'+SFweight+'*'+PromptGenLepMatch+'*'+METFilter_MC,
                    'FilesPerJob' : 5,
             }

ZZbaseW   = getBaseWnAOD(directory,'Fall2017_102X_nAODv7_Full2017v7',['ZZTo4L'])
#gg4mbaseW   = getBaseWnAOD(directory,'Fall2017_102X_nAODv5_Full2017v6',['ggZZ4m','ggZZ4m_ext1','ggZZ4m_ext2'])
#gg2e2mbaseW = getBaseWnAOD(directory,'Fall2017_102X_nAODv5_Full2017v6',['ggZZ2e2m','ggZZ2e2m_ext1'])
gg2m2tbaseW = getBaseWnAOD(directory,'Fall2017_102X_nAODv7_Full2017v7',['ggZZ2m2t_ext1'])

addSampleWeight(samples,'ZZ','ZZTo4L',        "1.07*"+ZZbaseW+"/baseW") ## The non-ggZZ NNLO/NLO k-factor, cited from https://arxiv.org/abs/1405.2219v1
#addSampleWeight(samples,'ZZ','ZZTo4L_ext1',   "1.07*"+ZZbaseW+"/baseW") 
#addSampleWeight(samples,'ZZ','ZZTo4L_ext2',   "1.07*"+ZZbaseW+"/baseW") 
addSampleWeight(samples,'ZZ','ZZTo2L2Nu',     "1.07")
addSampleWeight(samples,'ZZ','ZZTo2L2Q',      "1.07")
addSampleWeight(samples,'ZZ','ggZZ2e2t_ext1', "1.68") ## The NLO/LO k-factor, cited from https://arxiv.org/abs/1509.06734v1
#addSampleWeight(samples,'ZZ','ggZZ2m2t',      "1.68*"+gg2m2tbaseW+'/baseW') 
addSampleWeight(samples,'ZZ','ggZZ2m2t_ext1', "1.68*"+gg2m2tbaseW+'/baseW')
addSampleWeight(samples,'ZZ','ggZZ2e2m_ext1', "1.68")
#addSampleWeight(samples,'ZZ','ggZZ4m_ext2',   "1.68")


samples['VVV']  = {  'name'   :   getSampleFilesNano(directory,'ZZZ')
                                + getSampleFilesNano(directory,'WZZ')
                                + getSampleFilesNano(directory,'WWZ')
                                + getSampleFilesNano(directory,'WWW'),
                    'weight' : XSWeight+'*'+SFweight+'*'+PromptGenLepMatch+'*'+METFilter_MC ,
                  }



############ AZH SIGNAL SAMPLES ############
samples['AZH'] = {  'name': ['###/eos/user/s/srudrabh/AZH/postprocessing/fall17_102X_nAODv7/Fall2017_102X_nAODv7_Full2017v7/MCl1loose2017v7__MCCorr2017v7__l2loose__l2tightOR2017v7/nanoLatino_AZH_mA1000_mH600_private__part0.root',], 
                    'weight' : XSWeight+'*'+SFweight+'*'+PromptGenLepMatch+'*'+METFilter_MC ,
                  }

################## FAKE ###################
###########################################

samples['Fake']  = {   'name': [ ] ,
                       'weight' : fakeW+'*'+METFilter_DATA,
                       'weights' : [ ] ,
                       'isData': ['all'],
                       'FilesPerJob' : 500 ,
                       'suppressNegativeNuisances' :['all'],
                     }

directory = treeBaseDir+'Run2017_102X_nAODv7_Full2017v7/DATAl1loose2017v7__l2loose__fakeW'
for Run in DataRun :
  for DataSet in DataSets :
    FileTarget = getSampleFilesNano(directory,DataSet+'_'+Run[1],True)
    for iFile in FileTarget:
      samples['Fake']['name'].append(iFile)
      samples['Fake']['weights'].append(DataTrig[DataSet])

samples['Fake']['subsamples'] = {
    'e': 'abs(ZH3l_pdgid_l) == 11',
    'm': 'abs(ZH3l_pdgid_l) == 13'
}

###########################################
################## DATA ###################
###########################################

samples['DATA']  = {   'name': [ ] ,
                       'weight' : METFilter_DATA+'*'+LepWPCut,
                       'weights' : [ ],
                       'isData': ['all'],
                       'FilesPerJob' : 500 ,
                    }

directory = treeBaseDir+'Run2017_102X_nAODv7_Full2017v7/DATAl1loose2017v7__l2loose__l2tightOR2017v7'
for Run in DataRun :
  for DataSet in DataSets :
    FileTarget = getSampleFilesNano(directory,DataSet+'_'+Run[1],True)
    for iFile in FileTarget:
        samples['DATA']['name'].append(iFile)
        samples['DATA']['weights'].append(DataTrig[DataSet])

