import numpy as np

FILE = [
    'Untreated/Banfora/Banfora_UT_rep1_savedresults_PPv72.mat',
    'Untreated/Banfora/Banfora_UT_rep2_savedresults_PPv72.mat',
    'Untreated/Banfora/Banfora_UT_rep3_savedresults_PPv72.mat',
    'Untreated/Banfora/Banfora_UT_rep4_savedresults_PPv72.mat',

    'Untreated/Kisumu/Kisumu_UT_rep1_savedresults_PPv72.mat',
    'Untreated/Kisumu/Kisumu_UT_rep2_savedresults_PPv72.mat',
    'Untreated/Kisumu/Kis_UT_rep3_savedresults_PPv72.mat',
    'Untreated/Kisumu/Kisumu_UT_rep4_savedresults_PPv72.mat',
    'Untreated/Kisumu/Kis_UT_rep5_savedresults_PPv72.mat',

    'Untreated/Ngoussu/Ngoussu_UT_rep1_savedresults_PPv72.mat',
    'Untreated/Ngoussu/Ngoussu_UT_rep2_savedresults_PPv72.mat',
    'Untreated/Ngoussu/Ngoussu_UT_rep3_savedresults_PPv72.mat',
    'Untreated/Ngoussu/Ngoussu_UT_rep4_savedresults_PPv72.mat',

    'Untreated/VK7/VK7_UT_rep1_savedresults_PPv72.mat',
    'Untreated/VK7/VK7_UT_rep2_savedresults_PPv72.mat',
    'Untreated/VK7/VK7_UT_rep3_savedresults_PPv72.mat',
    'Untreated/VK7/VK7_UT_rep4_savedresults_PPv72.mat',


    'Olyset/Banfora/Tracks_Banfora_Olyset_rep 1.mat',
    'Olyset/Banfora/Tracks_Banfora_Rep2_olyset_pos_clean_compatible2.mat',
    'Olyset/Banfora/BAN_Oly_rep3_postprocessing.mat',
    'Olyset/Banfora/BAN_Oly_rep4_postprocessing.mat',
    'Olyset/Banfora/BAN_Oly_rep5_postprocessing.mat',
    'Olyset/Banfora/BAN_Oly_rep6_postprocessing.mat',

    'Olyset/Kisumu/Tracks_Kis_rep1_oly.mat',
    'Olyset/Kisumu/Tracks_Kis_rep2_oly.mat',
    'Olyset/Kisumu/Tracks_Kis_rep3_oly.mat',
    'Olyset/Kisumu/Tracks_rep 4 kisumu.mat',
    'Olyset/Kisumu/Tracks_Kis_oly_rep5.mat',
    'Olyset/Kisumu/Tracks_kis_rep6_oly.mat',

    'Olyset/Ngoussu/NG_Oly_Rep 1_postprocessing.mat',
    'Olyset/Ngoussu/Tracks_NG_Oly_Rep2_2.mat',
    'Olyset/Ngoussu/Tracks_NG_Oly_rep3_2.mat',
    'Olyset/Ngoussu/Tracks_NG_Rep4_oly.mat',
    'Olyset/Ngoussu/NG_Oly_Rep5_postprocessing.mat',
    'Olyset/Ngoussu/Tracks_NG_Rep6_oly.mat',

    'Olyset/VK7/Tracks_VK7_Rep1_oly.mat',
    'Olyset/VK7/Tracks_VK7_rep2_oly.mat',
    'Olyset/VK7/Tracks_VK7_Rep3_oly.mat',
    'Olyset/VK7/Tracks_VK7_Rep5_oly.mat',
    'Olyset/VK7/Tracks_VK7_oly_rep6.mat'
]

IS_RESISTANT = np.array([
    0,0,0,0,
    0,0,0,0,0,
    0,0,0,0,
    0,0,0,0,

    1,1,1,1,1,1,
    1,1,1,1,1,1,
    1,1,1,1,1,1,
    1,1,1,1,1
])

DATA_PATH = "E:/ITNS/olyset-vs-untreated/raw-data/" 
PATH = "E:/ITNS/olyset-vs-untreated/"