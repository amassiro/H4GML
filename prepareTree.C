
void prepareTree(std::string input_file_name = "data.root") {
    
    
    TFile* inputFile = new TFile (input_file_name.c_str(), "READ");   
    TTree* inputTree = (TTree*) ((TTree*) inputFile -> Get ("H4GSel")) -> Clone ("H4GSel");
    
    std::cout << " entries = " << inputTree->GetEntries() << std::endl;
    
    Double_t        p12_mass;
    Double_t        p13_mass;
    Double_t        p14_mass;
    Double_t        p23_mass;
    Double_t        p24_mass;
    Double_t        p34_mass;
    
    Double_t        dipho1_true_mass;
    Double_t        dipho2_true_mass;
    
   
    inputTree->SetBranchAddress("p12_mass", & p12_mass );
    inputTree->SetBranchAddress("p13_mass", & p13_mass );
    inputTree->SetBranchAddress("p14_mass", & p14_mass );
    inputTree->SetBranchAddress("p23_mass", & p23_mass );
    inputTree->SetBranchAddress("p24_mass", & p24_mass );
    inputTree->SetBranchAddress("p34_mass", & p34_mass );
 
    inputTree->SetBranchAddress("dipho1_true_mass", & dipho1_true_mass);
    inputTree->SetBranchAddress("dipho2_true_mass", & dipho2_true_mass);
    
    
    std::string output_file_name = input_file_name + ".train.root";
    
    
    TFile* outputFile = new TFile (output_file_name.c_str(), "RECREATE");   
    TTree *outputTree = inputTree->CloneTree(0);
    
    int best_combination; 
    outputTree->Branch("best_combination",  &best_combination, "best_combination/I");
    
    
    
    for (int iEntry=0; iEntry<inputTree->GetEntries(); iEntry++) {
      if (!(iEntry%50000)) std::cout << "   " << iEntry << " :: "  << inputTree->GetEntries() << std::endl;
      inputTree->GetEntry(iEntry);
      
      best_combination = -1;
      
      // 1-4, 2-3  ----> A
      float min_mass_A_1 = fabs (p14_mass - dipho1_true_mass) + fabs( p23_mass - dipho2_true_mass);
      float min_mass_A_2 = fabs (p14_mass - dipho2_true_mass) + fabs( p23_mass - dipho1_true_mass);
      
      float min_mass_A = min_mass_A_1 < min_mass_A_2 ? min_mass_A_1 : min_mass_A_2;
      
      
      // 1-3, 2-4  ----> B
      float min_mass_B_1 = fabs (p13_mass - dipho1_true_mass) + fabs( p24_mass - dipho2_true_mass);
      float min_mass_B_2 = fabs (p13_mass - dipho2_true_mass) + fabs( p24_mass - dipho1_true_mass);
      
      float min_mass_B = min_mass_B_1 < min_mass_B_2 ? min_mass_B_1 : min_mass_B_2;
      
      
      // 1-2, 3-4  ----> C      
      float min_mass_C_1 = fabs (p12_mass - dipho1_true_mass) + fabs( p34_mass - dipho2_true_mass);
      float min_mass_C_2 = fabs (p12_mass - dipho2_true_mass) + fabs( p34_mass - dipho1_true_mass);
      
      float min_mass_C = min_mass_C_1 < min_mass_C_2 ? min_mass_C_1 : min_mass_C_2;
      
      
      if ((min_mass_A <= min_mass_B) && (min_mass_A <= min_mass_C)) best_combination = 0;  // A
      if ((min_mass_B <= min_mass_A) && (min_mass_B <= min_mass_C)) best_combination = 1;  // B
      if ((min_mass_C <= min_mass_A) && (min_mass_C <= min_mass_B)) best_combination = 2;  // C
      
      outputTree->Fill();     
      
    }
    
    outputTree->AutoSave();
    
    outputFile->Close();
    
    std::cout << " out :    " << output_file_name << std::endl;
    
}

