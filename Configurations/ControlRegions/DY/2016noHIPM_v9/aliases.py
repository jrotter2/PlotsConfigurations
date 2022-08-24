import os
import copy
import inspect

configurations = os.path.realpath(inspect.getfile(inspect.currentframe())) # this file
configurations = os.path.dirname(configurations) # ggH2016
configurations = os.path.dirname(configurations) # Differential
configurations = os.path.dirname(configurations) # Configurations
configurations = os.path.dirname(configurations) # Configurations

#aliases = {}


mc = [skey for skey in samples if skey not in ('Fake', 'DATA', 'Dyemb')]
mc_emb = [skey for skey in samples if skey not in ('Fake', 'DATA')]

# LepCut2l__ele_mvaFall17V2Iso_WP90__mu_cut_Tight80x
eleWP = 'mvaFall17V2Iso_WP90'
muWP  = 'cut_Tight80x'

aliases['LepWPCut'] = {
    'expr': 'LepCut2l__ele_'+eleWP+'__mu_'+muWP,
    'samples': mc_emb + ['DATA']
}

aliases['LepWPSF'] = {
    'expr': 'LepSF2l__ele_'+eleWP+'__mu_'+muWP,
    'samples': mc_emb
}


# aliases['gstarLow'] = {
#     'expr': 'Gen_ZGstar_mass >0 && Gen_ZGstar_mass < 4',
#     'samples': 'VgS'
# }

# aliases['gstarHigh'] = {
#     'expr': 'Gen_ZGstar_mass <0 || Gen_ZGstar_mass > 4',
#     'samples': 'VgS'
# }

# aliases['embedtotal'] = {
#     'expr': 'embed_total_mva16',  # wrt. eleWP
#     'samples': 'Dyemb'
# }


# # Fake leptons transfer factor
# aliases['fakeW'] = {
#     'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP,
#     'samples': ['Fake']
# }

# # And variations - already divided by central values in formulas !
# aliases['fakeWEleUp'] = {
#     'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_EleUp',
#     'samples': ['Fake']
# }
# aliases['fakeWEleDown'] = {
#     'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_EleDown',
#     'samples': ['Fake']
# }
# aliases['fakeWMuUp'] = {
#     'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_MuUp',
#     'samples': ['Fake']
# }
# aliases['fakeWMuDown'] = {
#     'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_MuDown',
#     'samples': ['Fake']
# }
# aliases['fakeWStatEleUp'] = {
#     'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_statEleUp',
#     'samples': ['Fake']
# }
# aliases['fakeWStatEleDown'] = {
#     'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_statEleDown',
#     'samples': ['Fake']
# }
# aliases['fakeWStatMuUp'] = {
#     'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_statMuUp',
#     'samples': ['Fake']
# }
# aliases['fakeWStatMuDown'] = {
#     'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_statMuDown',
#     'samples': ['Fake']
# }

aliases['nCleanGenJet'] = {
    'linesToAdd': ['.L %s/src/PlotsConfigurations/Configurations/Differential/ngenjet.cc+' % os.getenv('CMSSW_BASE')],
    'class': 'CountGenJet',
    'samples': mc
}

# gen-matching to prompt only (GenLepMatch2l matches to *any* gen lepton)
aliases['PromptGenLepMatch2l'] = {
    'expr': 'Alt$(Lepton_promptgenmatched[0]*Lepton_promptgenmatched[1], 0)',
    'samples': mc
}

aliases['Top_pTrw'] = {
    'expr': '(topGenPt * antitopGenPt > 0.) * (TMath::Sqrt((0.103*TMath::Exp(-0.0118*topGenPt) - 0.000134*topGenPt + 0.973) * (0.103*TMath::Exp(-0.0118*antitopGenPt) - 0.000134*antitopGenPt + 0.973))) * (TMath::Sqrt(TMath::Exp(1.61468e-03 + 3.46659e-06*topGenPt - 8.90557e-08*topGenPt*topGenPt) * TMath::Exp(1.61468e-03 + 3.46659e-06*antitopGenPt - 8.90557e-08*antitopGenPt*antitopGenPt))) + (topGenPt * antitopGenPt <= 0.)', # Same Reweighting as other years, but with additional fix for tune CUET -> CP5
    'samples': ['top']
}


# ##### DY Z pT reweighting
# aliases['getGenZpt_OTF'] = {
#     'linesToAdd':['.L %s/src/PlotsConfigurations/Configurations/patches/getGenZpt.cc+' % os.getenv('CMSSW_BASE')],
#     'class': 'getGenZpt',
#     'samples': ['DY']
# }
# handle = open('%s/src/PlotsConfigurations/Configurations/patches/DYrew30.py' % os.getenv('CMSSW_BASE'),'r')
# exec(handle)
# handle.close()
# aliases['DY_NLO_pTllrw'] = {
#     'expr': '('+DYrew['2016']['NLO'].replace('x', 'getGenZpt_OTF')+')*(nCleanGenJet == 0)+1.0*(nCleanGenJet > 0)',
#     'samples': ['DY']
# }
# aliases['DY_LO_pTllrw'] = {
#     'expr': '('+DYrew['2016']['LO'].replace('x', 'getGenZpt_OTF')+')*(nCleanGenJet == 0)+1.0*(nCleanGenJet > 0)',
#     'samples': ['DY']
# }


# Jet bins
# using Alt$(CleanJet_pt[n], 0) instead of Sum$(CleanJet_pt >= 30) because jet pt ordering is not strictly followed in JES-varied samples

# No jet with pt > 30 GeV
aliases['zeroJet'] = {
    'expr': 'Alt$(CleanJet_pt[0], 0) < 30.'
}

aliases['oneJet'] = {
    'expr': 'Alt$(CleanJet_pt[0], 0) > 30.'
}

aliases['multiJet'] = {
    'expr': 'Alt$(CleanJet_pt[1], 0) > 30.'
}

aliases['SFweight2lAlt'] = {
    'expr'   : 'puWeight*TriggerSFWeight_2l*Lepton_RecoSF[0]*Lepton_RecoSF[1]*EMTFbug_veto',
    'samples': mc
}

aliases['trigger_bits'] = {
    'expr'    : 'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL \
                 || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL \
                 || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ \
                 || HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ \
                 || HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL \
                 || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL \
                 || HLT_IsoTkMu24 \
                 || HLT_IsoMu24 \
                 || HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ \
                 || HLT_Ele27_WPTight_Gsf \
                 || HLT_Ele25_eta2p1_WPTight_Gsf',
    'samples' : mc
}

# data/MC scale factors
aliases['SFweight'] = {
    # 'expr': ' * '.join(['SFweight2lAlt', 'LepWPCut', 'LepWPSF','PrefireWeight','Jet_PUIDSF_loose']),
    'expr': ' * '.join(['LepWPCut']), # , 'LepWPSF']),
    'samples': mc
}

# # variations
# aliases['SFweightEleUp'] = {
#     'expr': 'LepSF2l__ele_'+eleWP+'__Up',
#     'samples': mc_emb
# }
# aliases['SFweightEleDown'] = {
#     'expr': 'LepSF2l__ele_'+eleWP+'__Do',
#     'samples': mc_emb
# }
# aliases['SFweightMuUp'] = {
#     'expr': 'LepSF2l__mu_'+muWP+'__Up',
#     'samples': mc_emb
# }
# aliases['SFweightMuDown'] = {
#     'expr': 'LepSF2l__mu_'+muWP+'__Do',
#     'samples': mc_emb
# }

# # In WpWmJJ_EWK events, partons [0] and [1] are always the decay products of the first W
# aliases['lhe_mW1'] = {
#     'expr': 'TMath::Sqrt(2. * LHEPart_pt[0] * LHEPart_pt[1] * (TMath::CosH(LHEPart_eta[0] - LHEPart_eta[1]) - TMath::Cos(LHEPart_phi[0] - LHEPart_phi[1])))',
#     'samples': ['WWewk']
# }

# # and [2] [3] are the second W
# aliases['lhe_mW2'] = {
#     'expr': 'TMath::Sqrt(2. * LHEPart_pt[2] * LHEPart_pt[3] * (TMath::CosH(LHEPart_eta[2] - LHEPart_eta[3]) - TMath::Cos(LHEPart_phi[2] - LHEPart_phi[3])))',
#     'samples': ['WWewk']
# }

