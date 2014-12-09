% Computational Vision
% Practicum Face Recognition
%
% Student name: ...
%
% >> OBJECTIVE: 
% 1) Analize the code
% 2) Understand the function of each part of the code
% 3) Code the missing parts
% 4) Execute the code and check the results

% main function
function main()

clc; close all; clear;

%% These sub-directories are required
addpath(genpath('feature_extraction'))
addpath(genpath('classification'))

%% Load database of images and analyze the structure
ARFace = importdata('ARFace.mat');

%% Prepare the data set samples identifying data and labels (male/female).
% We will use the internal faces loaded in the structure
display(ARFace)

% 1. To complete:
% Answer these questions: 
% a. Why the size of the field internal, size(ARFace.internal), is 1188 x
% 2210?
% 2210 is the numbre of instances (images of faces) and
% 1188 is all the contained pixels at each image (33px x 36px).
%
% b. Which is the information contained in ARFace.person?
% The label corresponding at which person the instance
% belong.


%% Count the number of samples and samples males and females of the data set.
% This information is in ARFace.gender ==> male == 1, female == 0
NumberMales = sum(ARFace.gender);
NumberSamples = length(ARFace.gender);
NumberFemales = NumberSamples - NumberMales;
% 2. To complete:
% NumberMales = 988
% NumberSamples = 2210
% NumberFemales = 1222

%% Visualize some of the internal faces and save in bmp images
% Use the function reshape to transform the information from a vector to a
% matrix.
% 3. To complete:
accI = [];
row=[];
for i=1:20:NumberSamples
    I = mat2gray(reshape(ARFace.internal(:,i),36,33));
    row = cat(2, row, I);
    if mod(((i+19)/20),10)==0
        accI= cat(1, accI, row);
        row=[];
    end
end
figure;
imshow(accI);
imwrite(accI,'faces.jpg');

%% Define the training set and test set from the data set using:
% a. Use the whole data set (an unbalanced problem)
% Build this data structure: 
%   images(:,i) is the image of sample i.
%   labels(i) is the label of sample i.
%   subjects(i) is the number of the subject of sample i.
% Use the "internal" images, we will reduce dimensionality later.
images = ARFace.internal;
labels = ARFace.gender;
subjects = ARFace.person;

% 4. To complete:
% images = ...
% labels = ...
% subjects = ...
    
%% Atention! We will use the dataset in the representation: Sample x Variables (Samples x 1188):
images = images';
labels = labels';
subjects = subjects';


%% Feature Extraction using PCA
mat_features_pca = feature_extraction('PCA', images);


%% Feature Extraction using PCA (95% variance explained)
mat_features_pca95 = feature_extraction('PCA95', images);


%% Feature Extraction using LDA
mat_features_lda = feature_extraction('LDA', images, labels);


%% Classification
% Call the function validation to perform the F-fold
% cross validation with: the samples, labels, information
% about the training set subjects and F the number of folds.
F = 10;
Rates_pca = validation(mat_features_pca', labels', subjects', F);
display(Rates_pca);
Rates_pca95 = validation(mat_features_pca95', labels', subjects', F);
display(Rates_pca95);
Rates_lda = validation(mat_features_lda', labels', subjects', F);
display(Rates_lda);

% 11. To complete:
% Answer this question: 
% Which is the best result?
% Looking at Rates of pca, pca95 and lda, the best one is lda. It can be
% said because classify better extracting features using the lda method.
% It makes sense because pca and lda are both linear projection
% dimensionality reduction methods but lda is supervised. Seems logic that
% using labels the method is able to extract better projected dimensions of
% the data.

end



