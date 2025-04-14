# Machine Learning with Amex

This script uses the classical as well ase new methods of machine learning in order to do a data analysis of the American Express transactions downloaded from their website of Germany.

The methods implemented are:
- Neural Network: RNN
- Lasso Regression
- Ridge Regression
- PCA Regression
- Partial Least Squares
- Auto Regression
- Linear Regression



## Usage

- Dowload the data from the website of the Amex Germany (more info at the end of the instructions)
 ```
https://www.americanexpress.com/de-de/account/login?inav=iNLogBtn

Note: a dummy Excel data named "data.xlsx" is uploaded along with the scripts
```
- save the Excel data as data.xlsx into the folder of script

- open the script main.py

- the following required variables have to be changed according to your interest:

  --- **var**: which category are you interested in. For example if the data of "LIDL" is of your concern, then type "LIDL" or "REWE" or "Bahn" ...etc. However if you are interested of the whole data please write "all"
  
   --- **x_columns**: as a list. Here is to provide the list of predictors
  
   --- **y_column**: Here is to provide the column name of the response

   --- **lag**: the lagging time in days

depening on the performance of your laptop the following code my take some time in order to run all the methods and get the results of **test MSEs and y hats**
```
MSEs,y_preds=merging_all.m_all(df,x_columns, y_column, lag)
```
The following code is showing the **test MSEs** for all the methods sorted from low to high
```
MSEs
```
- Method: in order to present the results of interest we would need to choose a method from all of those implemented for example "PCA"

After running the code till line 77 the following results appear:
```
y hat and y values in order to do a comparison between the two
```

## Results

The last part of the script is do the plotting of **the variable** we have chosen for example "REWE" for the years available as well as **the regression line** we have got from the method specified above.

The results shoud appear as in the photo below:

![alt text](https://github.com/Anjrini/ML_Amex/blob/main/Pics/Results.png?raw=true)

>Note:
- The **lag** variable is required in order to implement the RNN data Frame for the methods to run as otherwise it comes down to only a simple linear regression
- The **ISLP Library**  is required and has to be downloaded using:
``` pip install ISLP ```


## Instructions on dowloading the data from the amex website

- First visit the website:
```https://www.americanexpress.com/de-de/account/login?inav=iNLogBtn```
- Sign in into the website and follow the steps below:

![alt text](https://github.com/Anjrini/ML_Amex/blob/main/Pics/instruct1.png?raw=true)

![alt text](https://github.com/Anjrini/ML_Amex/blob/main/Pics/instruct2.png?raw=true)

![alt text](https://github.com/Anjrini/ML_Amex/blob/main/Pics/instruct3.png?raw=true)

Should you have any query, kindly contact me.

Best regards,

Mustafa Anjrini
