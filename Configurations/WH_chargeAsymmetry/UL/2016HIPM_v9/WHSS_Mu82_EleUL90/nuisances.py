# nuisances
# name of samples here must match keys in samples.py 

from LatinoAnalysis.Tools.commonTools import getSampleFiles, getBaseW, addSampleWeight

def nanoGetSampleFiles(inputDir, Sample):
    return getSampleFiles(inputDir, Sample, False, 'nanoLatino_')

try:
    mc = [skey for skey in samples if skey != 'DATA' and not skey.startswith('Fake')]
except NameError:
    mc = []
    cuts = {}
    nuisances = {}
    def makeMCDirectory(x=''):
        return ''

from LatinoAnalysis.Tools.HiggsXSection import HiggsXSection
HiggsXS = HiggsXSection()


################################ EXPERIMENTAL UNCERTAINTIES  #################################

#### Luminosity

# https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2#LumiComb
# Uncorrelated 2016               1.0
# Uncorrelated       2017              2.0
# Uncorrelated             2018             1.5
# Correlated   2016, 2017, 2018   0.6, 0.9, 2.0
# Correlated         2017, 2018        0.6, 0.2

nuisances['lumi_Uncorrelated'] = {
    'name': 'lumi_13TeV_2016',
    'type': 'lnN',
    'samples': dict((skey, '1.010') for skey in mc if skey not in ['WZ'])
}

nuisances['lumi_Correlated_Run2'] = {
    'name': 'lumi_13TeV_Run2',
    'type': 'lnN',
    'samples': dict((skey, '1.006') for skey in mc if skey not in ['WZ'])
}


#### FAKES
nuisances['fake_syst_mm'] = {
    'name': 'CMS_fake_syst_mm',
    'type': 'lnN',
    'samples': {
        'Fake_mm': '1.3'
    },
}
nuisances['fake_syst_em'] = {
    'name': 'CMS_fake_syst_em',
    'type': 'lnN',
    'samples': {
        'Fake_em': '1.3'
    },
}
nuisances['fake_syst_ee'] = {
    'name': 'CMS_fake_syst_ee',
    'type': 'lnN',
    'samples': {
        'Fake_ee': '1.3'
    },
}

nuisances['fake_ele'] = {
    'name': 'CMS_fake_e_2016',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Fake': ['fakeWEleUp', 'fakeWEleDown'],
    }
}
nuisances['fake_ele_stat'] = {
    'name': 'CMS_fake_stat_e_2016',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Fake': ['fakeWStatEleUp', 'fakeWStatEleDown']
    }
}
nuisances['fake_mu'] = {
    'name': 'CMS_fake_m_2016',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Fake': ['fakeWMuUp', 'fakeWMuDown'],
    }   
}       
nuisances['fake_mu_stat'] = {
    'name': 'CMS_fake_stat_m_2016',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Fake': ['fakeWStatMuUp', 'fakeWStatMuDown'],
    }
}

###### B-tagger

for shift in ['jes', 'lf', 'hf', 'hfstats1', 'hfstats2', 'lfstats1', 'lfstats2', 'cferr1', 'cferr2']:
    btag_syst = ['(btagSF%sup)/(btagSF)' % shift, '(btagSF%sdown)/(btagSF)' % shift]

    name = 'CMS_btag_%s' % shift
    if 'stats' in shift:
        name += '_2016'

    nuisances['btag_shape_%s' % shift] = {
        'name'   : name,
        'kind'   : 'weight',
        'type'   : 'shape',
        'samples': dict((skey, btag_syst) for skey in mc),
    }

##### Trigger Scale Factors

trig_syst = ['SFtriggUp', 'SFtriggDown']

nuisances['trigg'] = {
    'name'   : 'CMS_eff_hwwtrigger_2016',
    'kind'   : 'weight',
    'type'   : 'shape',
    'samples': dict((skey, trig_syst) for skey in mc)
}

##### Electron Efficiency and energy scale

nuisances['eff_e'] = {
    'name'   : 'CMS_eff_e_2016',
    'kind'   : 'weight',
    'type'   : 'shape',
    'samples': dict((skey, ['SFweightEleUp', 'SFweightEleDown']) for skey in mc)
}

nuisances['eff_ttHMVA_e'] = {
    'name'   : 'CMS_eff_ttHMVA_e_2016',
    'kind'   : 'weight',
    'type'   : 'shape',
    'samples': dict((skey, ['LepWPttHMVASFEleUp', 'LepWPttHMVASFEleDown']) for skey in mc)
}

nuisances['electronpt'] = {
    'name'       : 'CMS_scale_e_2016',
    'kind'       : 'suffix',
    'type'       : 'shape',
    'mapUp'      : 'ElepTup',
    'mapDown'    : 'ElepTdo',
    'samples'    : dict((skey, ['1', '1']) for skey in mc),
    'folderUp'   : makeMCDirectory('ElepTup_suffix'),
    'folderDown' : makeMCDirectory('ElepTdo_suffix'),
    'AsLnN'      : '1'
}

##### Muon Efficiency and energy scale

nuisances['eff_m'] = {
    'name'   : 'CMS_eff_m_2016',
    'kind'   : 'weight',
    'type'   : 'shape',
    'samples': dict((skey, ['SFweightMuUp', 'SFweightMuDown']) for skey in mc)
}

nuisances['eff_ttHMVA_m'] = {
    'name'    : 'CMS_eff_ttHMVA_m_2016',
    'kind'    : 'weight',
    'type'    : 'shape',
    'samples' : dict((skey, ['LepWPttHMVASFMuUp', 'LepWPttHMVASFMuDown']) for skey in mc)
}

nuisances['muonpt'] = {
    'name'       : 'CMS_scale_m_2016',
    'kind'       : 'suffix',
    'type'       : 'shape',
    'mapUp'      : 'MupTup',
    'mapDown'    : 'MupTdo',
    'samples'    : dict((skey, ['1', '1']) for skey in mc),
    'folderUp'   : makeMCDirectory('MupTup_suffix'),
    'folderDown' : makeMCDirectory('MupTdo_suffix'),
    'AsLnN'      : '1'
}

##### Jet energy scale
jes_systs = ['JESAbsolute','JESAbsolute_2016','JESBBEC1','JESBBEC1_2016','JESEC2','JESEC2_2016','JESFlavorQCD','JESHF','JESHF_2016','JESRelativeBal','JESRelativeSample_2016']

for js in jes_systs:

  nuisances[js] = {
      'name'      : 'CMS_scale_' + js,
      'kind'      : 'suffix',
      'type'      : 'shape',
      'mapUp'     : js + 'up',
      'mapDown'   : js + 'do',
      'samples'   : dict((skey, ['1', '1']) for skey in mc),
      'folderUp'  : makeMCDirectory('RDF__JESup_suffix'),
      'folderDown': makeMCDirectory('RDF__JESdo_suffix'),
      'AsLnN'     : '1'
  }

##### Jet energy resolution
nuisances['JER'] = {
    'name'       : 'CMS_res_j_2016',
    'kind'       : 'suffix',
    'type'       : 'shape',
    'mapUp'      : 'JERup',
    'mapDown'    : 'JERdo',
    'samples'    : dict((skey, ['1', '1']) for skey in mc),
    'folderUp'   : makeMCDirectory('JERup_suffix'),
    'folderDown' : makeMCDirectory('JERdo_suffix'),
    'AsLnN'      : '1'
}

##### MET unclustered energy

# metUp.PuppiMET_pt_METup
nuisances['met'] = {
    'name'      : 'CMS_scale_met_2016',
    'kind'      : 'suffix',
    'type'      : 'shape',
    'mapUp'     : 'METup',
    'mapDown'   : 'METdo',
    'samples'   : dict((skey, ['1', '1']) for skey in mc),
    'folderUp'  : makeMCDirectory('METup_suffix'),
    'folderDown': makeMCDirectory('METdo_suffix'),
    'AsLnN'     : '1'
}


##### Pileup

# puWeight_UL2016
nuisances['PU'] = {
    'name': 'CMS_PU_2016',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'DY'      : ['1.008421*(puWeightUp/puWeight)', '0.992494*(puWeightDown/puWeight)'],
        'WW'      : ['1.010428*(puWeightUp/puWeight)', '0.990400*(puWeightDown/puWeight)'],
        'ggWW'    : ['1.012367*(puWeightUp/puWeight)', '0.988402*(puWeightDown/puWeight)'],
        'Vg'      : ['0.995630*(puWeightUp/puWeight)', '1.007076*(puWeightDown/puWeight)'],
        'WZ'      : ['1.002037*(puWeightUp/puWeight)', '0.997721*(puWeightDown/puWeight)'],
        'ZZ'      : ['1.006574*(puWeightUp/puWeight)', '0.993175*(puWeightDown/puWeight)'],
        'VVV'     : ['1.012827*(puWeightUp/puWeight)', '0.989623*(puWeightDown/puWeight)'],
        'top'     : ['1.009336*(puWeightUp/puWeight)', '0.991324*(puWeightDown/puWeight)'],
        'Higgs'   : ['1.011609*(puWeightUp/puWeight)', '0.989039*(puWeightDown/puWeight)'],
    },
    'AsLnN': '1',
}

### PU ID SF uncertainty

puid_syst = ['Jet_PUIDSF_loose_up/Jet_PUIDSF_loose', 'Jet_PUIDSF_loose_down/Jet_PUIDSF_loose']

nuisances['jetPUID'] = {
    'name': 'CMS_PUID_2016',
    'kind': 'weight',
    'type': 'shape',
    'samples': dict((skey, puid_syst) for skey in mc)
}

### PS and UE

nuisances['PS_ISR']  = {
    'name'    : 'PS_ISR',
    'kind'    : 'weight',
    'type'    : 'shape',
    'samples' : dict((skey, ['PSWeight[2]', 'PSWeight[0]']) for skey in mc),
    'AsLnN'   : '1',
}

nuisances['PS_FSR']  = {
    'name'    : 'PS_FSR',
    'kind'    : 'weight',
    'type'    : 'shape',
    'samples' : dict((skey, ['PSWeight[3]', 'PSWeight[1]']) for skey in mc),
    'AsLnN'   : '1',
}

nuisances['UE_CP5']  = {
    'name'    : 'UE_CP5',
    'skipCMS' : 1,
    'type'    : 'lnN',
    'samples' : dict((skey, '1.015') for skey in mc),
}

# Charge flip SF
nuisances['chargeFlip'] = {
    'name': 'CMS_whss_chargeFlip_2016',
    'kind': 'weight',
    'type': 'shape',
    'samples' : dict((skey, ['1-ttHMVA_SF_err_flip_2l[0]', '1+ttHMVA_SF_err_flip_2l[0]']) for skey in ['top','WW']),
    'cuts'    : [cut for cut in cuts if ('_ee_' in cut)]
}

# Charge flip efficiency
nuisances['chargeFlipEff'] = {
    'name'    : 'CMS_whss_chargeFlipEff_2016',
    'kind'    : 'weight',
    'type'    : 'shape',
    'samples' : dict((skey, ['1-ttHMVA_eff_err_flip_2l[0]', '1+ttHMVA_eff_err_flip_2l[0]']) for skey in ['DY']),
    'cuts'    : [cut for cut in cuts if ('_ee_' in cut)]
}

####### Generic "cross section uncertainties"

apply_on = {
    'top': [
        'isSingleTop * 1.0816 + isTTbar',
        'isSingleTop * 0.9184 + isTTbar'
    ]
}

nuisances['singleTopToTTbar'] = {
    'name'    : 'singleTopToTTbar',
    'skipCMS' : 1,
    'kind'    : 'weight',
    'type'    : 'shape',
    'samples' : apply_on
}

## Top pT reweighting uncertainty

nuisances['TopPtRew'] = {
    'name'       : 'CMS_topPtRew',   # Theory uncertainty
    'kind'       : 'weight',
    'type'       : 'shape',
    'samples'    : {
        'top': ["1.", "1./Top_pTrw"]
    },
    'symmetrize' : True
}

nuisances['WgStar'] = {
    'name'    : 'CMS_hww_WgStarScale',
    'type'    : 'lnN',
    'samples' : {
        'WgS' : '1.25'
    }
}

###### pdf uncertainties

valuesggh  = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ggH', '125.09','pdf','sm')
valuesggzh = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ggZH','125.09','pdf','sm')
valuesbbh  = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','bbH', '125.09','pdf','sm')

nuisances['pdf_Higgs_gg'] = {
    'name': 'pdf_Higgs_gg',
    'samples': {
        'ggH_hww': valuesggh,
        'ggH_htt': valuesggh,
        'ggZH_hww': valuesggzh,
        'bbH_hww': valuesbbh
    },
    'type': 'lnN',
}

values = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ttH','125.09','pdf','sm')

nuisances['pdf_Higgs_ttH'] = {
    'name': 'pdf_Higgs_ttH',
    'samples': {
        'ttH_hww': values
    },
    'type': 'lnN',
}

valuesqqh = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','vbfH','125.09','pdf','sm')
valueswh  = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','WH',  '125.09','pdf','sm')
valueszh  = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ZH',  '125.09','pdf','sm')

nuisances['pdf_Higgs_qqbar'] = {
    'name': 'pdf_Higgs_qqbar',
    'type': 'lnN',
    'samples': {
        'qqH_hww'     : valuesqqh,
        'qqH_htt'     : valuesqqh,
        'WH_hww_plus' : valueswh,
        'WH_hww_minus': valueswh,
        'WH_htt_plus' : valueswh,
        'WH_htt_minus': valueswh,
        'ZH_hww'      : valueszh,
        'ZH_htt'      : valueszh
    },
}

nuisances['pdf_qqbar'] = {
    'name': 'pdf_qqbar',
    'type': 'lnN',
    'samples': {
        'Wg': '1.04',
        'Zg': '1.04',
        'ZZ': '1.04',  # PDF: 0.0064 / 0.1427 = 0.0448493
        'WZ': '1.04',  # PDF: 0.0064 / 0.1427 = 0.0448493
        'WgS': '1.04', # PDF: 0.0064 / 0.1427 = 0.0448493
        'ZgS': '1.04', # PDF: 0.0064 / 0.1427 = 0.0448493
    },
}

nuisances['pdf_Higgs_gg_ACCEPT'] = {
    'name': 'pdf_Higgs_gg_ACCEPT',
    'samples': {
        'ggH_hww': '1.006',
        'ggH_htt': '1.006',
        'ggZH_hww': '1.006',
        'bbH_hww': '1.006'
    },
    'type': 'lnN',
}
nuisances['pdf_gg_ACCEPT'] = {
    'name': 'pdf_gg_ACCEPT',
    'samples': {
        'ggWW': '1.006',
    },
    'type': 'lnN',
}

nuisances['pdf_Higgs_qqbar_ACCEPT'] = {
    'name': 'pdf_Higgs_qqbar_ACCEPT',
    'type': 'lnN',
    'samples': {
        'qqH_hww'     : '1.002',
        'qqH_htt'     : '1.002',
        'WH_hww_plus' : '1.003',
        'WH_hww_minus': '1.003',
        'WH_htt_plus' : '1.003',
        'WH_htt_minus': '1.003',
        'ZH_hww'      : '1.002',
        'ZH_htt'      : '1.002',
    },
}

nuisances['pdf_qqbar_ACCEPT'] = {
    'name': 'pdf_qqbar_ACCEPT',
    'type': 'lnN',
    'samples': {
        'ZZ': '1.001',
        'WZ': '1.001',
    },
}

##### Renormalization & factorization scales

## Shape nuisance due to QCD scale variations for DY
# LHE scale variation weights (w_var / w_nominal)

## This should work for samples with either 8 or 9 LHE scale weights (Length$(LHEScaleWeight) == 8 or 9)
variations = ['LHEScaleWeight[0]', 'LHEScaleWeight[1]', 'LHEScaleWeight[3]', 'LHEScaleWeight[Length$(LHEScaleWeight)-4]', 'LHEScaleWeight[Length$(LHEScaleWeight)-2]', 'LHEScaleWeight[Length$(LHEScaleWeight)-1]']

# nuisances['QCDscale_V'] = {
#     'name'    : 'QCDscale_V',
#     'skipCMS' : 1,
#     'kind'    : 'weight_envelope',
#     'type'    : 'shape',
#     'samples' : {
#         'DY': variations
#     },
#     'AsLnN'   : '1'
# }

nuisances['QCDscale_VV'] = {
    'name'    : 'QCDscale_VV',
    'kind'    : 'weight_envelope',
    'type'    : 'shape',
    'samples' : {
        'WW'  : variations,
        'Zg'  : variations,
        'Wg'  : variations,
        'ZZ'  : variations,
        'WZ'  : variations,
        'WgS' : variations,
        'ZgS' : variations
    }
}

nuisances['QCDscale_ggVV'] = {
    'name' : 'QCDscale_ggVV',
    'type' : 'lnN',
    'samples': {
        'ggWW': '1.15',
    },
}

#### QCD scale uncertainties for Higgs signals other than ggH

values = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','vbfH','125.09','scale','sm')

nuisances['QCDscale_qqH'] = {
    'name': 'QCDscale_qqH',
    'samples': {
        'qqH_hww': values,
        'qqH_htt': values
    },
    'type': 'lnN'
}

valueswh = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','WH','125.09','scale','sm')
valueszh = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ZH','125.09','scale','sm')

nuisances['QCDscale_VH'] = {
    'name': 'QCDscale_VH',
    'samples': {
        'WH_hww_plus' : valueswh,
        'WH_hww_minus': valueswh,
        'WH_htt_plus' : valueswh,
        'WH_htt_minus': valueswh,
        'ZH_hww'      : valueszh,
        'ZH_htt'      : valueszh
    },
    'type': 'lnN',
}

values = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ggZH','125.09','scale','sm')

nuisances['QCDscale_ggZH'] = {
    'name': 'QCDscale_ggZH',
    'samples': {
        'ggZH_hww': values
    },
    'type': 'lnN',
}

values = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ttH','125.09','scale','sm')

nuisances['QCDscale_ttH'] = {
    'name': 'QCDscale_ttH',
    'samples': {
        'ttH_hww': values
    },
    'type': 'lnN',
}

nuisances['QCDscale_WWewk'] = {
    'name': 'QCDscale_WWewk',
    'samples': {
        'WWewk': '1.11',
    },
    'type': 'lnN'
}

nuisances['QCDscale_qqbar_ACCEPT'] = {
    'name': 'QCDscale_qqbar_ACCEPT',
    'type': 'lnN',
    'samples': {
        'qqH_hww'     : '1.003',
        'qqH_htt'     : '1.003',
        'WH_hww_plus' : '1.010',
        'WH_hww_minus': '1.010',
        'WH_htt_plus' : '1.010',
        'WH_htt_minus': '1.010',
        'ZH_hww'      : '1.015',
        'ZH_htt'      : '1.015',
    }
}

#FIXME: these come from HIG-16-042, maybe should be recomputed?
nuisances['QCDscale_gg_ACCEPT'] = {
    'name': 'QCDscale_gg_ACCEPT',
    'samples': {
        'ggH_htt'  : '1.012',
        'ggH_hww'  : '1.012',
        'ggZH_hww' : '1.012',
        'ggWW'     : '1.012',
    },
    'type': 'lnN',
}

# WZ normalization from control region

nuisances['WZ2jnorm']  = {
    'name'    : 'CMS_hww_WZ3l2jnorm',
    'samples' : {
        'WZ' : '1.00',
    },
    'type' : 'rateParam',
    'cuts' : [cut for cut in cuts if '2j' in cut],
}

nuisances['WZ1jnorm']  = {
    'name'    : 'CMS_hww_WZ3l1jnorm',
    'samples' : {
        'WZ' : '1.00',
    },
    'type' : 'rateParam',
    'cuts' : [cut for cut in cuts if '1j' in cut],
}

## Use the following if you want to apply the automatic combine MC stat nuisances.
nuisances['stat']  = {
    'type'          : 'auto',
    'maxPoiss'      : '10',
    'includeSignal' : '1',
    'samples'       : {}
}
#  nuisance ['maxPoiss'] =  Number of threshold events for Poisson modelling
#  nuisance ['includeSignal'] =  Include MC stat nuisances on signal processes (1=True, 0=False)

for n in nuisances.values():
    n['skipCMS'] = 1
