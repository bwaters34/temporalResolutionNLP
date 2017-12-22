Enclosed is most of the code for our project "Temporal Resolution of Texts through X-Bar Theory of Phrase Structure."

Group Members: Peter DelMastro, Roy Jackman, Brennan Waters

*****

Because this project incorporated a lot of code to build the dataset, we were unable to include all of the source files. The entire project is available on Github at https://github.com/bwaters34/temporalResolutionNLP.

*****

The enclosed code pertains to running the actual training and testing of the model. It will not work when run because the training data could also not be included.

*****

The root folder contains the code to train our implemented NaiveBayes model on the Gutenberg and Proquest Datasets using any set of desired features. These features need to be changed manually, which is intuitive enough to do. Running 

python run_me.py

will run this file. Currently it is set to "trees" as features. The other options as "unigrams", "bigrams", and "pos". Such features were stored as deserialized defaultdicts in the *Dataset/Train and *Dataset/Test directories (see Github for the whole project directory structure).

Other important files in the root include NaiveBayes.py (our implementation), validation.py (for cross-validation and plots), and tokenize.py (for reading from the train and test set files).

*****

The SKLearn folder contains files for training any SKLearn classification model on a large sparse matrix, found in *Dataset/Train/Numpy and *Dataset/Test/Numpy where * can be either Gutenberg and Proquest. (Note: To reiterate, these folders are not included). Again, the selected features must be entered manually, and 

python run_me.py

in the SKLearn directory will train and test the model on the selected features given the selected machine learning model. It is currently set to LogReg and "trees."

*****

The Clustering Directory contains files for both creating the word-cluster datasets and training a model on them. Running

python create_dataset.py

is used to create the dataset, given that such data is present. NaiveBayes and BinClassifier are classification models adapted with work with numpy arrays as features as to allow for sklearn models to be used as well. Again, validation.py is for training and evaluation. Running

python run_me.py

will training and eval a Logistic Regression classifier on the clustering data.