MIR @ CISUC - http://mir.dei.uc.pt
Renato Panda - panda@dei.uc.pt

-----------------------------------
Feature set used in our TAFFC paper[1]

* all_features.csv:
  File containing all features in comma-separated values (CSV) format.
  The first line contains the header describing each column, first column
  is the "IDSong" (name of the clip), the remaining are the feature names.
  The first 1603 features (F0001 to F1603) are the standard/baseline ones,
  as described in the paper.
  After those, all the others are the novel proposed features as described
  in the same article. The name of these should be clear, e.g.,
  "ORIGINAL-EXPRESSIVE_TECHNIQUES-Vibrato Presence" is "Vibrato presence",
  i.e., if vibrato was found in the clip, which belongs to expressive
  techniques features, extracted from the original audio clip.

* all_features_decorrelated.csv:
  This is similar to the previous file but without the features identified
  as "highly correlated", as described. This was the set of features we
  used in our feature selection and classification experiments and thus
  I advise you to use it, since the previous one contains duplicated info
  (e.g., MFCCs extracted with different audio frameworks).

* top100_features.csv:  
  This is a subset of the previous file containing only the top 100 features, 
  ordered from most important to least important. 


* features.csv:
  The standard features are only identified as "Fxxxx" (feature code) in the
  feature matrix. This file describes these features if more information is
  needed (e.g., name, toolbox used, category and so on).



1 - Panda R., Malheiro R. & Paiva R. P. (2018). “Novel audio features for 
music emotion recognition”. IEEE Transactions on Affective Computing (IEEE 
early access). DOI: 10.1109/TAFFC.2018.2820691.
http://mir.dei.uc.pt/pdf/Journals/MOODetector/TAFFC_2018_Panda.pdf
