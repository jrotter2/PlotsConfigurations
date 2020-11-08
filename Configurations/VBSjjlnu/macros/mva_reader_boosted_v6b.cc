#include "LatinoAnalysis/MultiDraw/interface/TTreeFunction.h"
#include "LatinoAnalysis/MultiDraw/interface/FunctionLibrary.h"
#include "NNEvaluation/DNNTensorflow/interface/DNNEvaluator.hh"

#include "TFile.h"
#include "TMath.h"
#include "TGraph.h"

#include <cmath>
#include <string>
#include <iostream>

using namespace std;
using namespace NNEvaluation;

#ifndef MVAREADERBoosted
#define MVAREADERBoosted

typedef TTreeReaderValue<Double_t> DoubleValueReader;

class MVAReaderBoosted : public multidraw::TTreeFunction {
public:
  
  MVAReaderBoosted(const char* model_path,  const char* transform_path, bool verbose, int category);

  char const* getName() const override { return "MVAReaderBoosted"; }
  TTreeFunction* clone() const override { return new MVAReaderBoosted(model_path_.c_str(), 
                                           transform_path_.c_str(), verbose, category_); }

  std::string model_path_;
  std::string transform_path_;
  int category_;
  TGraph * dnn_transformation; 
  unsigned getNdata() override { return 1; }
  double evaluate(unsigned) override;

protected:  
 
  bool verbose;
  void bindTree_(multidraw::FunctionLibrary&) override;
  ~MVAReaderBoosted();
  
  DNNEvaluator* dnn_tensorflow;

  IntValueReader* VBS_category{};

  FloatArrayReader* Lepton_pt{};
  FloatArrayReader* Lepton_eta{};
  DoubleValueReader* nJets30{};

  FloatValueReader* vbs_0_pt{};
  FloatValueReader* vbs_1_pt{};
  FloatValueReader* vjet_0_pt{};
  FloatValueReader* vjet_0_eta{};

  FloatValueReader* mjj_vbs{};
  FloatValueReader* mjj_vjet{};

  FloatValueReader* deltaeta_vbs{};
  FloatValueReader* deltaphi_vbs{};
  
  FloatValueReader* Zvjets_0{};
  FloatValueReader* Zlep{};

  FloatValueReader* Asym_vbs{};
  FloatValueReader* A_ww{};
  FloatValueReader* Centr_vbs{};
  FloatValueReader* Centr_ww{};

  DoubleValueReader* vbs_0_qgl_boost{};
  DoubleValueReader* vbs_1_qgl_boost{};

};


MVAReaderBoosted::MVAReaderBoosted(const char* model_path, const char* transform_path, bool verbose, int category):
    model_path_(model_path), 
    transform_path_(transform_path),
    verbose(verbose),
    category_(category)
{
    dnn_tensorflow = new DNNEvaluator(model_path_, verbose);

    // Load the TGRaph used to transform the DNN score
    // The TGraph is the cumulative distribution of the DNN on the signal
    TFile * tf_file = new TFile(transform_path_.c_str(), "READ");
    dnn_transformation = (TGraph*) tf_file->Get("cumulative_signal");
    tf_file->Close();
}

MVAReaderBoosted::~MVAReaderBoosted(){
  
  delete dnn_transformation;
  delete dnn_tensorflow;
}


double
MVAReaderBoosted::evaluate(unsigned)
{
  // Run only if 
  if ( *(VBS_category->Get()) != category_) {
    return -999.;
  }


  // Filter out some events
  if (*(mjj_vbs->Get()) < 500 ) return -999;
  if (*(deltaeta_vbs->Get()) < 2.5 ) return -999;
  if (*(vbs_0_pt->Get()) < 50) return -999;
  if (*(vbs_1_pt->Get()) < 30) return -999;

  std::cout << "Preparing inputs" <<endl;

  std::vector<float> input{};
  input.push_back( Lepton_pt->At(0) );
  input.push_back( Lepton_eta->At(0) );
  input.push_back( (float) *(nJets30->Get()) );
  input.push_back( *(vbs_0_pt->Get()) );
  input.push_back( *(vbs_1_pt->Get()) );
  input.push_back( *(vjet_0_pt->Get()) );
  input.push_back( *(vjet_0_eta->Get()) );
  input.push_back( *(mjj_vbs->Get()) );
  input.push_back( *(mjj_vjet->Get()) );
  input.push_back( *(deltaeta_vbs->Get()) );
  input.push_back( *(deltaphi_vbs->Get()) );
  input.push_back( *(Zvjets_0->Get()) );
  input.push_back( *(Zlep->Get()) );
  input.push_back( *(Asym_vbs->Get()) );
  input.push_back( *(A_ww->Get()) );
  input.push_back( *(Centr_vbs->Get()) );
  input.push_back( *(Centr_ww->Get()) );
  input.push_back( (float) *(vbs_0_qgl_boost->Get()) );
  input.push_back( (float) *(vbs_1_qgl_boost->Get()) );
  
  vector<float> dnn_scores = dnn_tensorflow->analyze(input);
  return dnn_transformation->Eval(dnn_scores.at(0));
}

void
MVAReaderBoosted::bindTree_(multidraw::FunctionLibrary& _library)
{  
  _library.bindBranch(VBS_category, "VBS_category");
  _library.bindBranch(Lepton_pt, "Lepton_pt");
  _library.bindBranch(Lepton_eta, "Lepton_eta");
  _library.bindBranch(nJets30, "nJets30");

  _library.bindBranch(vbs_0_pt, "vbs_0_pt");
  _library.bindBranch(vbs_1_pt, "vbs_1_pt");
  _library.bindBranch(vjet_0_pt, "vjet_0_pt");
  _library.bindBranch(vjet_0_eta, "vjet_0_eta");
 
  _library.bindBranch(mjj_vbs, "mjj_vbs");
  _library.bindBranch(mjj_vjet, "mjj_vjet");

  _library.bindBranch(deltaeta_vbs, "deltaeta_vbs");
  _library.bindBranch(deltaphi_vbs, "deltaphi_vbs");
  
  _library.bindBranch(Zvjets_0, "Zvjets_0");
  _library.bindBranch(Zlep, "Zlep");

  _library.bindBranch(Asym_vbs, "Asym_vbs");
  _library.bindBranch(A_ww, "A_ww");
  _library.bindBranch(Centr_vbs, "Centr_vbs");
  _library.bindBranch(Centr_ww, "Centr_ww");

  _library.bindBranch(vbs_0_qgl_boost, "vbs_0_qgl_boost");
  _library.bindBranch(vbs_1_qgl_boost, "vbs_1_qgl_boost");


}


#endif 