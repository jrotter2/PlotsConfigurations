# structure configuration for datacard
structure ={}

# keys here must match keys in samples.py 
for iproc in samples.keys():
    structure[iproc] = {
        'isSignal' : 1 if 'WW_B' in iproc else 0,
        'isData'   : 1 if iproc == 'DATA' else 0,
    }

structure['WW_B10']['removeFromCuts'] = ['ww2l2v_13TeV_sr_2j_B0']
structure['WW_B2']['removeFromCuts'] = ['ww2l2v_13TeV_sr_2j_B7']
structure['WW_B9']['removeFromCuts'] = ['ww2l2v_13TeV_sr_2j_B0']
structure['Vg']['removeFromCuts'] = ['ww2l2v_13TeV_sr_2j_B9','ww2l2v_13TeV_sr_2j_B1']

for nuis in nuisances.itervalues():
  if 'cutspost' in nuis:
    nuis['cuts'] = nuis['cutspost'](nuis, cuts)

    print nuis
