import os

resubs = [
    #'Wjets_NLOnj.0',
    #'Wjets_NLOnj.2',
    #'Wjets_NLOnj.3',
    #'Wjets_NLOnj.4',
    #'Wjets_NLOnj.6',
    #'Wjets_NLOnj.8',
    #'Wjets_NLOnj.13',
    #'Wjets_NLOnj.14',
    #'Wjets_NLOnj.20',
    #'Wjets_NLOnj.24',
    #'Wjets_NLOptM.43',
    #'Wjets_NLOptMRaw.2',
    #'Wjets_NLOptMRaw.3',
    #'Wjets_NLOptMRaw.15',
    #'Wjets_NLOptMRaw.19',
    #'Wjets_NLOptMRaw.20',
    #'Wjets_NLOptMRaw.21',
    #'Wjets_NLOptMRaw.22',
    #'Wjets_NLOptMRaw.23',
    #
    #'Wjets_NLOstatM.0',
    #'Wjets_NLOstatM.1',
    #'Wjets_NLOstatM.2',
    #'Wjets_NLOstatM.3',
    #'Wjets_NLOstatM.4',
    #'Wjets_NLOstatM.5',
    #'Wjets_NLOstatM.6',
    #'Wjets_NLOstatM.7',
    #'Wjets_NLOstatM.8',
    #'Wjets_NLOstatM.9',
    #'Wjets_NLOstatM.10',
    #'Wjets_NLOstatM.11',
    #'Wjets_NLOstatM.12',
    #'Wjets_NLOstatM.13',
    #'Wjets_NLOstatM.14',
    #'Wjets_NLOstatM.15',
    #'Wjets_NLOstatM.16',
    #'Wjets_NLOstatM.17',
    #'Wjets_NLOstatM.18',
    #'Wjets_NLOstatM.19',
    #'Wjets_NLOstatM.20',
    #'Wjets_NLOstatM.21',
    #'Wjets_NLOstatM.22',
    #'Wjets_NLOstatM.23',
    #'Wjets_NLOstatM.24',
    #'Wjets_NLOstatM.25',
    #'Wjets_NLOstatM.26',
    #'Wjets_NLOstatM.27',
    #'Wjets_NLOstatM.28',
    #'Wjets_NLOstatM.29',
    #'Wjets_NLOstatM.30',
    #'Wjets_NLOstatM.31',
    #'Wjets_NLOstatM.32',
    #'Wjets_NLOstatM.33',
    #'Wjets_NLOstatM.34',
    #'Wjets_NLOstatM.35',
    #'Wjets_NLOstatM.36',
    #'Wjets_NLOstatM.37',
    #'Wjets_NLOstatM.38',
    #'Wjets_NLOstatM.39',
    #'Wjets_NLOstatM.40',
    #'Wjets_NLOstatM.41',
    #'Wjets_NLOstatM.42',
    #'Wjets_NLOstatM.43',
    #'Wjets_NLOstatM.44',
    #'Wjets_NLOstatM.45',
    #'Wjets_NLOstatM.46',
    #'Wjets_NLOstatM.47',
    #'Wjets_NLOstatM.48',
    #'Wjets_NLOstatM.49',
    #'Wjets_NLOstatM.50',
    #'Wjets_NLOstatM.51',
    #'Wjets_NLOstatM.52',
    #'Wjets_NLOstatM.53',
    #'Wjets_NLOstatM.54',
    #'Wjets_NLOstatM.55',
    'Wjets_NLOstatM.56',
    #'Wjets_NLOstatM.57',
    #'Wjets_NLOstatM.58',
    #'Wjets_NLOstatM.59',
    #'Wjets_NLOstatM.60',
    #'Wjets_NLOstatM.61',
    #'Wjets_NLOstatM.62',
    #'Wjets_NLOstatM.63',
    #'Wjets_NLOstatM.64',
    #'Wjets_NLOstatM.65',
    #'Wjets_NLOstatM.66',
    #'Wjets_NLOstatM.67',
    #'Wjets_NLOstatM.68',
    #'Wjets_NLOstatM.69',
    #'Wjets_NLOstatM.70',
    #'Wjets_NLOstatM.71',
    #'Wjets_NLOstatM.72',
    #'Wjets_NLOstatM.73',
    #'Wjets_NLOstatM.74',
    #'Wjets_NLOstatM.75',
    #'Wjets_NLOstatM.76',
    #'Wjets_NLOstatM.77',
    #'Wjets_NLOstatM.78',
    #'Wjets_NLOstatM.79',
    #'Wjets_NLOstatM.80',
    #'Wjets_NLOstatM.81',
    #'Wjets_NLOstatM.82',
    #'Wjets_NLOstatM.83',
    #'Wjets_NLOstatM.84',
    #'Wjets_NLOstatM.85',
    #'Wjets_NLOstatM.86',
]


resub_str = 'ssh m8 qsub -l walltime=168:00:00 -N mkShapes__Wjets_inv_2018v7__ALL__JOB -q localgrid@cream02 -o /user/svanputt/monoHiggs/sl7/CMSSW_10_6_5/src/job//mkShapes__Wjets_inv_2018v7__ALL/JOB/mkShapes__Wjets_inv_2018v7__ALL__JOB.out -e /user/svanputt/monoHiggs/sl7/CMSSW_10_6_5/src/job//mkShapes__Wjets_inv_2018v7__ALL/JOB/mkShapes__Wjets_inv_2018v7__ALL__JOB.err /user/svanputt/monoHiggs/sl7/CMSSW_10_6_5/src/job//mkShapes__Wjets_inv_2018v7__ALL/JOB/mkShapes__Wjets_inv_2018v7__ALL__JOB_Sing.sh > /user/svanputt/monoHiggs/sl7/CMSSW_10_6_5/src/job//mkShapes__Wjets_inv_2018v7__ALL/JOB/mkShapes__Wjets_inv_2018v7__ALL__JOB.jid'

for job in resubs:
    comd = resub_str.replace('JOB', job)
    os.system(comd)

#ssh m8 qsub -l walltime=168:00:00 -N mkShapes__Wjets_inv_2018v7__ALL__JOB -q localgrid@cream02 -o /user/svanputt/monoHiggs/sl7/CMSSW_10_6_5/src/job//mkShapes__Wjets_inv_2018v7__ALL/JOB/mkShapes__Wjets_inv_2018v7__ALL__JOB.out -e /user/svanputt/monoHiggs/sl7/CMSSW_10_6_5/src/job//mkShapes__Wjets_inv_2018v7__ALL/JOB/mkShapes__Wjets_inv_2018v7__ALL__JOB.err /user/svanputt/monoHiggs/sl7/CMSSW_10_6_5/src/job//mkShapes__Wjets_inv_2018v7__ALL/JOB/mkShapes__Wjets_inv_2018v7__ALL__JOB_Sing.sh > /user/svanputt/monoHiggs/sl7/CMSSW_10_6_5/src/job//mkShapes__Wjets_inv_2018v7__ALL/JOB/mkShapes__Wjets_inv_2018v7__ALL__JOB.jid
