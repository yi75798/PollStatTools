# PollStatTools

## Overview

PollStatTools is a set of tools for pollsters to analyze polling data.
It provides a set of functions to calculate polling statistics, especially the function of weighting.
You can use this package to replace some function involving weighting in those paid statistical softwares such as SPSS and Stata.

## Features
- Supports generation of weighted frequency distribution table.
- Supports generation of weighted cross tabulation.
- Supports generation of weighted average and standard deviation.
- Supports generation of weighted t-test.
- Supports generation of weighted one-way ANOVA.
- Supports generation of post-hoc test using Bonferroni correction.
- ...

## Installation
In order to use this package, make sure you have Python 3.9 and python packages as below:
- pandas
- numpy
- scipy
- statesmodels
- math
- decimal

Then, you can clone this repository by running the following command in the terminal:
```python
git clone https://github.com/yi75798/PollStatTools
cd PollStatTools
```

Finally, you can install the package by running the following command in your script:
```python
from PollStatTools.PollStatTools import *
```

## Usage
### rounding
***rounding(num, decimal=0)***

Round to the specified number of digits

#### Parameters
>**num : float, the number to be rounded**
        
>**decimal : int, optional, the number of decimal places to round to, default 0**

#### Returns
**float**
>The rounded number.

### freq
***freq(data, var, w=None, label=None, to_clipboard=False)***

Generate frequency table for a variable with weight values.

#### Parameters
>**data : pd.DataFrame, the source of dataframe with the variable to be analyzed**
        
>**var : str, the variable to be analyzed**
        
>**w : str, optional, the weighting variable, default None, means weight=1**
        
>**label : str, optional, the label for the values of analyzed variable, default None**
        
>**to_clipboard : bool, optional, whether to copy the result to the clipboard, default False**

#### Returns
**pandas.DataFrame**
>The  dataframe of frequency distribution table.

### cross
***cross(data, r_var, c_var, w=None, percent_by='r', to_clipboard=False)***

Generate cross tabulation for two variables with weight values.

#### Parameters
>**data : pd.DataFrame, the source of dataframe with the variables to be analyzed**
        
>**r_var : str, the variable to be analyzed in the first column**
        
>**c_var : str, the variable to be analyzed in the second column**
        
>**w : str, optional, the weighting variable, default None, means weight=1**
        
>**percent_by : str, optional, the variable to be used to calculate the percentage, default 'r'**
        
>**to_clipboard : bool, optional, whether to copy the result to the clipboard, default False**

#### Returns
**pandas.DataFrame**
>The dataframe of cross tabulation.

### weighted_AnS
***weighted_AnS(df, var, weight)***

Calculate the weighted average and standard deviation of a variable with weight values.

#### Parameters
>**df : pd.DataFrame, the source of dataframe with the variable to be analyzed**
        
>**var : str, the variable to be analyzed**
        
>**weight : str, the weighting variable**

#### Returns
**tuple**
>The tuple with weighted average and standard deviation.
>The first element is the weighted average, the second element is the standard deviation.

### t_test
***t_test(df, var, group1, group2, w=None, group1_label='group1', group2_label='group2', to_clipboard=False)***

Calculate the t-test for two groups with weight values.

#### Parameters
>**df : pd.DataFrame, the source of dataframe with the variable to be analyzed**
        
>**var : str, the variable to be analyzed**
        
>**group1 : list, the group1 of the variable. It must be a subset of the index of df**
        
>**group2 : list, the group2 of the variable**
        
>**w : str, optional, the weighting variable, default None, means weight=1**
        
>**group1_label : str, optional, the label for the group1, default 'group1'**
        
>**group2_label : str, optional, the label for the group2, default 'group2'**
        
>**to_clipboard : bool, optional, whether to copy the result to the clipboard, default False**

#### Returns
**pandas.DataFrame**
>The dataframe of t-test result for two groups.

### one_way_anova
***one_way_anova(df, dv, groupby, w=None, to_clipboard=False)***

Calculate the one-way ANOVA for a variable with weight values.

#### Parameters
>**df : pd.DataFrame, the source of dataframe with the variable to be analyzed**
        
>**dv : str, the variable to be analyzed**
        
>**groupby : str, the grouping variable**
        
>**w : str, optional, the weighting variable, default None, means weight=1**
        
>**to_clipboard : bool, optional, whether to copy the result to the clipboard, default False**

#### Returns
**pandas.DataFrame**
>The dataframe of one-way ANOVA result.

### post_hoc
***post_hoc(df, dv, groupby, w=None, alpha=0.05, to_clipboard=False)***

Calculate the post-hoc test using Bonferroni correction for a variable with weight values.

#### Parameters
>**df : pd.DataFrame, the source of dataframe with the variable to be analyzed**
        
>**dv : str, the variable to be analyzed**
        
>**groupby : str, the grouping variable**
        
>**w : str, optional, the weighting variable, default None, means weight=1**
        
>**alpha : float, optional, the significance level, default 0.05**
        
>**to_clipboard : bool, optional, whether to copy the result to the clipboard, default False**

#### Returns
**pandas.DataFrame**
>The dataframe of post-hoc test result.

## Contact
If you have any questions or suggestions, please feel free to contact me at yi75798@gmail.com.