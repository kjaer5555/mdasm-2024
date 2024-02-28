## 1.Introduction

In the following report we have presented our analysis of the seven diseases in the data set, the quality of the photos, and the evaluation of the disease distribution within the 100 masked photos we have generated. We have shown the similarities and differences between the diseases and we have included extra commentaries on the symptoms which help identify each disease in a separate file.

## 2.Diseases description

According to our research the diseases in the dataset can be divided into two major groups: pre-cancerous (ACK, SEK, NEV) and cancerous (SCC, MEL, BCC). Some of the lesions in the former group can lead to a cancerous condition from the second group and therefore it is essential to know what are the symptoms that can help us make a distinction between the diseases.  

The first of our findings is that we concluded a relation between ACK and SCC. The cause of the diseases is excessive exposure to UV radiation and their appearance is also similar – white skin patches that feel like sandpaper. The lesion can start as ACK and develop into SCC.  

The authors of the dataset have combined the Bowen’s disease (BOD) and the Squamous Cell Carcinoma (SCC) under the name of the latter one. This is because the BOD is also called SCC in situ. The term “in situ” refers to a surface type of skin cancer and if BOD is not being treated, it can progress to an invasive SCC. BOD is usually a red, scaly patch, which might itch, crust or ooze, but most have no particular feeling.

Another strong relation occurs between NEV and MEL. In most cases the former is considered as a birthmark and a benign tumor, and its presence is mostly based on genetics. However, if triggered it can evolve into a malignant infection and start spreading causing Melanoma.  

Medics haven't found a specific cause for SEK, but it is known that the disease is triggered by aging. They can vary in color and shape so as a result the tumors can hide the co-existence of another malignant lesion. This further intensifies the importance of correct diagnosis. 

About the final disease, BCC, we have not found a direct predecessor, but it can also co-exist with some of the other lesions in our dataset. The main cause for BCC is UV radiation but what makes it different is that it develops in areas that are congested with blood vessels.  
## 3.Data set analysis

### 3.1 Photo quality

Due to the fact that the photos have been taken with a smartphone device, their quality varies across the data set. About two thirds of the photos are either not focused well, or there is a reflection of the flash which has "burned" some pixels, making them completely white. On some there are hairs which interfere with the lesion, making it hard to determine its exact borders. And in some occasions the lesion has raised sections making it impossible for the camera to have the whole of it focused, resulting in a blur in certain sections.

### 3.2	Metadata analysis  

Our data exploration aimed to assess how patient clinical information
impacts skin cancer detection, focusing on eight clinical features.

The analysis revealed distinct patterns between pigmented (NEV, MEL, SEK)
and non-pigmented lesions (ACK, BCC, SCC), particularly in terms of bleeding
and pain. Notably, ACK lesions generally do not cause pain, contrasting with
the often-painful SCC and BCC lesions.

The investigation into itching and patient age further differentiated
lesion types. Pigmented lesions tended to itch more frequently than their
non-pigmented counterparts. Age analysis showed a lower median age for NEV
compared to MEL and SEK, suggesting age as a useful discriminator for these
conditions. However, age did not significantly vary for ACK, SCC, and BCC,
indicating its limited predictive value for non-pigmented lesions. 

Prevalence of certain skin lesions across body regions have been found,
highlighting specific patterns, such as ACK's frequency on the forearm and
the commonality of NEV, SCC, BCC, MEL, and SEK on the face. Additional
feature analysis showed MEL and ACK rarely exhibit skin elevation, a
characteristic that sets them apart from other lesions. Pattern changes
were mostly associated with MEL, an essential marker for detecting this
cancer type. Although it was challenging to discern clear trends for recent
size increases, ACK notably lacked this feature. 

## 4. Conclusion

After analyzing the data, we have concluded that the diseases are distributed in a similar manner as they are in the whole data set, but since some of them are present on as few as 3 photos, we should mask a larger number of photos for better results.

## 5. Citations
### Metadata analysis

The complete metadata analysis of the dataset -  
Andre G.C. Pacheco, Renato A. Krohling, 
The impact of patient clinical information on automated skin cancer detection. 
