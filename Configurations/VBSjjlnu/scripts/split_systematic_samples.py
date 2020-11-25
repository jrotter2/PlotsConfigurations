import ROOT as R
import sys

'''
This script can be used to rename a systematic shape including the name
of the samples. 

In practice it splits the nuisances between samples if originally 
they had the same shape name. 
'''

f = R.TFile(sys.argv[1], "UPDATE")
shape_name = sys.argv[2]

wjets_bins = {}
for ir in range(1,7):
    wjets_bins["Wjets_HT_res_"+str(ir)] = "res_"+str(ir)
for ir in range(1,6):
    wjets_bins["Wjets_HT_boost_"+str(ir)] =  "boost_"+str(ir)


samples= wjets_bins

f.ls()
for k in f.GetListOfKeys():
    print(k)
    R.gDirectory.Cd(k.GetName())
    for z in R.gDirectory.GetListOfKeys():
        R.gDirectory.Cd(z.GetName())
        for l in R.gDirectory.GetListOfKeys():

            for sample, target_name in samples.items():
                new_shape_name = shape_name +"_"+ target_name
                
                if "histo_"+sample+"_"+shape_name+"Down" == l.GetName():
                    print(l.GetName(),"histo_"+sample+"_"+shape_name+"Down" )
                    obj = R.gDirectory.Get(l.GetName())
                    obj.SetName("histo_" + sample+"_"+new_shape_name+"Down")
                    obj.Write()
                if "histo_"+sample+"_"+shape_name+"Up" == l.GetName():
                    print(l.GetName(),"histo_"+sample+"_"+shape_name+"Down" )
                    obj = R.gDirectory.Get(l.GetName())
                    obj.SetName("histo_" + sample+"_"+new_shape_name+"Up")
                    obj.Write()
            
        R.gDirectory.Cd("../")

    R.gDirectory.Cd("../")

#f.Write()
f.Close()
