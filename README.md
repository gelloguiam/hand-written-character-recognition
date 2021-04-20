## To run the scripts
1. Put the training data in raw_training folder, put the test data in raw_test folder
2. Run ```generate_testdata.py```, this will generate the training dataset, test dataset, and setup the test folders for the classification
3. Sort and label the training data and place in data_training folder
4. Segment the training data to training-validation, put the validation images in data_validation folder
5. Run ```retrain_inception.py```, this will export a .pb file or a keras model based on Inception-v3 architecture in inception folder
6. Run the ```classify_characters.py```, this will import the .pb file and export individual summary of the test images
