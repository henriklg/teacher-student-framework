# Teacher-student framework
By Henrik Gjestang as part of master degree in Computational Science: Imaging and Biomedical Computing.

## Outline
Medical data is growing at an estimated 2.5 exabytes per year. However, medical data is often sparse and unavailable for the research community, and qualified medical personnel rarely have time for the tedious labeling work required to prepare the data. 
New screening methods of the gastrointestinal (GI) tract, like video capsule endoscopy (VCE), can help to reduce patients discomfort and help to increase screening capabilities. One of the main reasons why VCE is not more commonly used by medical experts is the amount of data it produces. 
A high level of extra work is required by the physicians who, depending on the patient, have to look at more than 50,000 frames per examination. 
To make VCE more accepted and useful data analysis methods such as machine learning can be very useful. 

Even if a lot of frames are collected per patient they are most of the time showing normal tissue without any relevant finding. This introduces another problem, namely that it is difficult to train a machine learning based method using this data. 
Existing models often struggle with the challenge of not having enough data that contains anomalies. This often leads to overfitted and not generalisable models. Our work explores ways to help existing models to overcome this problem by utilising a popular sub-category of machine learning called semi-supervised learning.
Semi-supervised learning uses a combination of labeled and unlabeled data which allows us to take advantage of large amounts of unlabeled data.

In this thesis, we introduce our proposed semi-supervised teacher-student framework. This framework is built specifically to take advantage of vast amount of unlabeled data and consists of three main steps: (1) train a teacher model with labeled data, (2) use the teacher model to infer pseudo labels with unlabeled data, and (3) train a new and larger student model with a combination of labeled images and inferred pseudo labels. 
These three steps are repeated several times by treating the student as a teacher to relabel the unlabeled data and consequently training a new student.

We demonstrate that our framework can be of use for classifying both, VCE and endoscopic colonoscopy images or videos. We demonstrate that our teacher-student model can significantly increase the performance compared to traditional supervised-learning-based models. We believe that our framework has potential to be a useful addition to existing medical multimedia systems for automatic disease detection, because new data can be continuously added to improve the models performance while in production. 


## Requirements
Everything is run on Ubuntu 20.04 within a conda environment (see environment.yml for all dependencies).
