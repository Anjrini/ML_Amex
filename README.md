# Machine Learning with Amex

This script uses the classical as well as the new methods of machine learning in order to do a data analysis of the data of the American Express transactions downloaded from their website of Germany.

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

Note: a dummy Excel data named "data.xlsx" is uploaded along with the   scripts
```
- save the Excel data as data.xlsx into the folder of script

- open the script main.py

- the following required variables have to be changed according to your interest:
--- var: which category are you interested in. For example if the data of "LIDL" is of your concern, then type "LIDL" or "REWE" or "Bahn" ...etc. However if you are interested of the whole data please write "all"

- x_columns: as a list. Here is to provide the list of predictors

- y_column: Here is to provide the column name of the response

- lag: the lagging time in days

depening on the performance of your laptop the following code my take some time in order to run all the methods and get the results MSEs and y_preds
```
MSEs,y_preds=merging_all.m_all(df,x_columns, y_column, lag)
```
The following code is showing the MSEs for all the methods sorted from low to high
```
MSEs
```
- Method: in order to present the results of interest we would need to choose a method from all of those implemented for example "PCA"

After running the code till line 77 the following results appear:
```
y hat and y values in order to do a comparison between the two
```

The last part of the script is do the plotting of the variable we have chosen for example "REWE" for the years available as well as the y hat we have got from the method specified above.

The results shoud appear as in the photo below
