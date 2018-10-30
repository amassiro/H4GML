# H4GML

Problem:

    4 photons to be assigned to the two "a"


Prepare tree:

    r00t prepareTree.C\(\"data/signal_skim_m_10.root\"\)

    ls data/ | grep -v train | awk '{print "r00t -q prepareTree.C\\\(\\\"data/"$1"\\\"\\\)"}'

    hadd all.root data/signal_skim_m_*.train.root
    
    
Train

    python train.py
    
    