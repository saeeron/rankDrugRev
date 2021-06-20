from rankDrugRev import RankDrugRev

rev = RankDrugRev()

rev.load_data() 

rev.fit_preprocess()

transformed_seqs, rating, drug_OH, cond_OH, useC = rev.transform_preprocess()