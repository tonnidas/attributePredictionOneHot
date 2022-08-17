# Attribute Prediction OneHot

This is the python version of the [repository](https://github.com/sarwanpasha/Attriubute-Prediction-Code) by SarWwan Ali. It was initially written in R. 

Input: 
1. Graph edges list
2. Graph attribute list

Output: 
1. knn accuracy for all the attributes. 
2. svm accuracy for all the attributes. 
3. naive bias accuracy for all the attributes. 
4. decision tree accuracy for all the attributes. 
Here, Accuracy = correctly predicted labels / all labels of test set.


Repository running instruction: 
`attributePredictionOneHot.py` is the main python file. For loading the inputs, it uses `inputDataLoad.py` file and all the methods are written in `helperMethods.py` that `attributePredictionOneHot.py` uses. 

Requirements:
requirements.txt file has a list of all required items to be installed in VM.

How to run: 
After installing every needed libraris in VM of this repository, the below command will show all the accuracies for all attributes. 
```
python attributePredictionOneHot.py
```