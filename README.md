# A Machine Learning Approach for the Diagnosis of Parkinson's Disease via Speech Analysis

## Introduction

This project, researched in March 2022, aims to provide a machine learning-based approach for accurately diagnosing Parkinson's Disease using speech analysis. Due to the deletion of the initial account, the project was reuploaded.

Parkinson’s Disease, the second most prevalent neurodegenerative disorder globally, affects over 10 million people. The current diagnostic methods are only 53% accurate for early diagnosis (within 5 years of symptoms). This project explores a machine learning approach using a dataset from the University of Oxford, focusing on various speech features.

## Background

### Parkinson's Disease

Parkinson’s is characterized by the death of dopamine-containing cells in the substantia nigra, impacting motor and cognitive abilities. Symptoms include frozen facial features, slowness of movement, tremors, and voice impairment.

### Performance Metrics

- **Accuracy:** (TP+TN)/(P+N)
- **Matthews Correlation Coefficient:** 1=perfect, 0=random, -1=completely inaccurate

## Algorithms Employed

1. **Logistic Regression (LR):** Uses the sigmoid logistic equation.
2. **Linear Discriminant Analysis (LDA):** Assumes Gaussian data with the same variance.
3. **k Nearest Neighbors (KNN):** Predictions based on the k closest instances.
4. **Decision Tree (DT):** Binary tree structure for predictions.
5. **Neural Network (NN):** Models the human brain's decision-making.
6. **Naive Bayes (NB):** Assumes independence between features.
7. **Gradient Boost (GB):** Combines weak learners to create a strong learner.

## Engineering Goal

Produce a machine learning model for Parkinson’s diagnosis with at least 90% accuracy and/or a Matthews Correlation Coefficient of at least 0.9. Compare algorithms and parameters to determine the best model.

## Dataset Description

- **Source:** University of Oxford
- **195 instances:** 147 Parkinson's subjects, 48 without Parkinson's
- **22 features:** Characteristics like frequency, pitch, amplitude/period of the sound wave
- **1 label:** 1 for Parkinson’s, 0 for no Parkinson’s

## Summary of Procedure

1. Split dataset into training and validation sets.
2. Train each algorithm: LR, LDA, KNN, DT, NN, NB, GB.
3. Evaluate results using the validation set.
4. Repeat for different training/validation splits and a rescaled dataset.
5. Conduct 5 trials and average the results.

## Data Analysis

In general, the models performed best on the rescaled dataset with a 75-25 train-test split. K Nearest Neighbors and Neural Network achieved the highest accuracy of 98%.

## Conclusion and Significance

This project demonstrates that machine learning significantly improves Parkinson’s diagnosis compared to current methods. The models achieved 98% accuracy, a critical improvement for effective treatment.

## Future Research

- Creating a mobile application for voice-based Parkinson's diagnosis.
- Using larger datasets in conjunction with the University of Oxford dataset.
- Tuning and improving models further.
- Investigating different neural network structures.
- Constructing novel algorithms for Parkinson's prediction.

## References

- Brooks, Megan. "Diagnosing Parkinson's Disease Still Challenging." Medscape Medical News, National Institute of Neurological Disorders, 31 July 2014.
- Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. BioMedical Engineering OnLine 2007, 6:23 (26 June 2007).
- Hashmi, Sumaiya F. "A Machine Learning Approach to Diagnosis of Parkinson’s Disease." Claremont Colleges Scholarship, Claremont College, 2013.
- Ozcift, Akin, and Arif Gulten. "Classifier Ensemble Construction with Rotation Forest to Improve Medical Diagnosis Performance of Machine Learning Algorithms." Computer Methods and Programs in Biomedicine 104.3 (2011): 443-51.
- Salvatore, C., A. Cerasa, I. Castiglioni, F. Gallivanone, A. Augimeri, M. Lopez, G. Arabia, M. Morelli, M.c. Gilardi, and A. Quattrone. "Machine Learning on Brain MRI Data for Differential Diagnosis of Parkinson's Disease and Progressive Supranuclear Palsy." Journal of Neuroscience Methods 222 (2014): 230-37.
- Shahbakhi, Mohammad, Danial Taheri Far, and Ehsan Tahami. "Speech Analysis for Diagnosis of Parkinson’s Disease Using Genetic Algorithm and Support Vector Machine." Journal of Biomedical Science and Engineering 07.04 (2014): 147-56.
- Sriram, Tarigoppula V. S., M. Venkateswara Rao, G. V. Satya Narayana, and D. S. V. G. K. Kaladhar. "Diagnosis of Parkinson Disease Using Machine Learning and Data Mining Systems from Voice Dataset." SpringerLink. Springer, Cham, 01 Jan. 1970.
- Bind, Shubham. "A Survey of Machine Learning Based Approaches for Parkinson Disease Prediction." International Journal of Computer Science and Information Technologies 6 (2015).
- Karplus, Abraham. "Machine Learning Algorithms for Cancer Diagnosis." Mar. 2012.
