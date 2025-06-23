#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   Description_ana.py
# Time    :   2023/11/01 10:02:55
# Author  :   Hsu, Liang-Yi 
# Email:   yi75798@gmail.com
# Description : Statistics analysis tools for polling data

import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.weightstats import ttest_ind as weight_ttest
import math

## Rounding
class rounding:
    '''
    Round to the specified number of digits

    Parameters
    ----------
    num : float
        The number to be rounded    
    decimal : int, optional
        The number of decimal places to round to, by default 0

    Returns
    -------
    float
        The rounded number 
    '''

    def __new__(cls, num, decimal=0):
        str_deci = 1
        if decimal != 0:
            str_deci /= (10 ** decimal)
            str_deci = str(str_deci)
            result = Decimal(str(num)).quantize(Decimal(str_deci), rounding=ROUND_HALF_UP)
            return float(result)
        else:
            str_deci /= (10 ** 1)
            str_deci = str(str_deci)
            result = Decimal(str(num)).quantize(Decimal(str_deci), rounding=ROUND_HALF_UP)
            result = Decimal(str(result)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            return int(result)

### Frequency
class freq:
    '''
    Generated frequency table for a variable with weight values.

    Parameters
    ----------
    data : pd.DataFrame
        The source of dataframe with the variable to be analyzed
    var : str
        The variable to be analyzed
    w : str, optional
        The weighting variable, by default None, means weight=1
    label : str, optional
        The label for the values of analized variable, by default None
    to_clipboard : bool, optional
        Whether to copy the result to the clipboard, by default False
    
    Returns
    -------
    pandas.DataFrame
        The frequency table
    '''
    def __new__(cls, data, var: str, w=None, label=None, to_clipboard=False) -> pd.DataFrame:
        df = data
        if w:
            a = pd.Series(df[[var, w]].groupby(var).sum()[w]) / df[w].sum()
            if label:
                b = label
            else:
                b = a.index
            c = a.apply(rounding, args=(3,))*100
            d = df[[var, w]].groupby(var).sum()[w].apply(rounding, args=(0,))
            df_temp = pd.DataFrame({'Label': b, 'Num': d, 'Freq(%)': c})

        else:
            df['w'] = 1
            a = pd.Series(df[[var, 'w']].groupby(var).sum()['w']) / df['w'].sum()
            if label:
                b = label
            else:
                b = a.index
            c = a.apply(rounding, args=(3,))*100
            d = df[[var, 'w']].groupby(var).sum()['w'].apply(rounding, args=(0,))
            df_temp = pd.DataFrame({'Label': b, 'Num': d, 'Freq(%)': c})
        
        if to_clipboard:
            df_temp.to_clipboard()
            return df_temp
        else:
            return df_temp

### cross table
class cross:
    '''
    Generate cross table for two variables with weight values.

    Parameters
    ----------
    data : pd.DataFrame
        The source of dataframe with the variables to be analyzed
    r_var : str
        The variable to be analyzed in the first column
    c_var : str
        The variable to be analyzed in the second column
    w : str, optional
        The weighting variable, by default None, means weight=1
    percent_by : str, optional
        The variable to be used to calculate the percentage, by default 'r'
    to_clipboard : bool, optional
        Whether to copy the result to the clipboard, by default False

    Returns
    -------
    pandas.DataFrame
        The cross table
    '''
    def __new__(cls, data, r_var: str, c_var: str ,w=None, percent_by='r', to_clipboard=False):
        df = data
        if w:
            if percent_by == 'r':
                df_num = pd.crosstab(df[r_var], df[c_var], values=df[w], aggfunc=sum,
                                    margins=True).fillna(0).applymap(rounding, decimal=0)
                df_freq = pd.crosstab(df[r_var], df[c_var], values=df[w], aggfunc=sum,
                                        margins=True,
                                        normalize='index').fillna(0).applymap(rounding, decimal=4)*100
                df_cross = pd.merge(df_num, df_freq, on=r_var, suffixes=['_n', '_rfreq'])
                
            elif percent_by== 'c':
                df_num = pd.crosstab(df[r_var], df[c_var], values=df[w], aggfunc=sum,
                                        margins=True).fillna(0).applymap(rounding, decimal=0)
                df_freq = pd.crosstab(df[r_var], df[c_var],
                                        margins=True,
                                        normalize='columns').fillna(0).applymap(rounding, decimal=4)*100
                df_cross = pd.merge(df_num, df_freq, on=r_var, suffixes=['_n', '_cfreq'])
                
        else:
            if percent_by == 'r':
                df_num = pd.crosstab(df[r_var], df[c_var],
                                    margins=True).fillna(0).applymap(rounding, decimal=0)
                df_freq = pd.crosstab(df[r_var], df[c_var],
                                        margins=True,
                                        normalize='index').fillna(0).applymap(rounding, decimal=4)*100
                df_cross = pd.merge(df_num, df_freq, on=r_var, suffixes=['_n', '_rfreq'])
                
            elif percent_by== 'c':
                df_num = pd.crosstab(df[r_var], df[c_var],
                                    margins=True).fillna(0).applymap(rounding, decimal=0)
                df_freq = pd.crosstab(df[r_var], df[c_var],
                                        margins=True,
                                        normalize='columns').fillna(0).applymap(rounding, decimal=4)*100
                df_cross = pd.merge(df_num, df_freq, on=r_var, suffixes=['_n', '_cfreq'])
        
        if to_clipboard:
            df_cross.to_clipboard()
            return df_cross
        else:
            return df_cross

### t-test
class weighted_AnS:
    '''
    Calculate the weighted average and standard deviation of a variable with weight values.

    Parameters
    ----------
    df : pd.DataFrame
        The source of dataframe with the variable to be analyzed
    var : str
        The variable to be analyzed
    weight : str
        The weighting variable

    Returns
    -------
    tuple
        The weighted average and standard deviation.
        The first element is the weighted average, the second element is the standard deviation.
    '''
    def __new__(cls, df, var:str, weight: str):
        average = np.average(df[var], weights=df[weight])
    
        variance = np.average((df[var]-average)**2, weights=df[weight])
        return (average, math.sqrt(variance))

class t_test:
    '''
    Calculate the t-test for two groups with weight values.
    
    Parameters
    ----------
    df : pd.DataFrame
        The source of dataframe with the variable to be analyzed
    var : str
        The variable to be analyzed
    group1 : list
        The group1 of the variable.
        It must be a subset of the index of df
    group2 : list
        The group2 of the variable
        It must be a subset of the index of df
    w : str, optional
        The weighting variable, by default None, means weight=1
    group1_label : str, optional
        The label for the group1, by default 'group1'
    group2_label : str, optional
        The label for the group2, by default 'group2'
    to_clipboard : bool, optional
        Whether to copy the result to the clipboard, by default False

    Returns
    -------
    pandas.DataFrame
        The t-test result for two groups
    '''
    def __new__(cls, df, var:str, group1, group2, w=None, group1_label='group1', group2_label='group2', to_clipboard=False):
        #### 可能要加 F-test 檢查變異數是否相等
        g1 = df.loc[group1]
        g2 = df.loc[group2]

        if w:
            g1w = g1[w]
            g2w = g2[w]
            t_stat, p, dof = weight_ttest(g1[var], g2[var], weights=(g1w, g2w))
            result = pd.DataFrame({'group': [group1_label, group2_label],
                                'n': [rounding(g1[w].sum()), rounding(g2[w].sum())],
                                'Mean': [weighted_AnS(g1, var, w)[0], weighted_AnS(g2, var, w)[0]],
                                'Std': [weighted_AnS(g1, var, w)[1], weighted_AnS(g2, var, w)[1]],
                                't_stat': [t_stat, None],
                                'p-value': [p, None],
                                'df': [dof, None]})
        
        else:
            t_stat, p = stats.ttest_ind(g1[var], g2[var])
            result = pd.DataFrame({'group': [group1_label, group2_label],
                                'Mean': [g1[var].mean(), g2[var].mean()],
                                'Std': [g1[var].std(), g2[var].std()],
                                't_stat': [t_stat, None],
                                'p-value': [p, None]})
        
        if to_clipboard:
            result.to_clipboard()
            return result
        else:
            return result

### ANOVA test
class one_way_anova:
    '''
    Calculate the one-way ANOVA for a variable with weight values.

    Parameters
    ----------
    df : pd.DataFrame
        The source of dataframe with the variable to be analyzed
    dv : str
        The variable to be analyzed
    groupby : str
        The grouping variable
    w : str, optional
        The weighting variable, by default None, means weight=1
    to_clipboard : bool, optional
        Whether to copy the result to the clipboard, by default False

    Returns
    -------
    pandas.DataFrame    
        The one-way ANOVA result
    '''
    def __new__(cls, df, dv:str, groupby:str, w=None, to_clipboard=False):
        if w:
            ### MSB
            ## SSB
            total_mean = weighted_AnS(df, dv, w)[0]
            group = sorted(df[groupby].unique())
            group = [df.loc[df[groupby]==g] for g in group]
            #group = [np.array(g[dv]) for g in group]
            
            n = np.array([g[w].sum() for g in group])
            mean = np.array([np.average(g[dv], weights=g[w]) for g in group])
            m_t = [(m - total_mean)**2 for m in mean]
            ssb = np.sum(n * m_t)
            dfb = len(n) - 1
            msb = ssb/dfb

            ### MSW
            ## SSW
            var = np.array([np.average((g[dv]-m)**2, weights=g[w]) for g, m in zip(group,mean)])
            ssw = np.sum(var * n)
            dfw = df[w].sum() - len(n)
            msw = ssw/dfw

            ### F, p
            F_stat = msb/msw
            p_value = 1 - stats.f.cdf(F_stat, dfb, dfw)
        
        else:
            model = ols(f'{dv} ~ C({groupby})',df).fit()
            anovaResults =  anova_lm(model, typ=2)
            F_stat = anovaResults['F'].iloc[0]
            p_value = anovaResults['PR(>F)'].iloc[0]
        
        if to_clipboard:
            result = pd.DataFrame({'F_stat': [rounding(F_stat, 2)], 'p_value': [rounding(p_value, 3)]})
            result.to_clipboard()
            return result
        else:
            result = pd.DataFrame({'F_stat': [rounding(F_stat, 2)], 'p_value': [rounding(p_value, 3)]})
            return result

### post_hoc test
class post_hoc:
    '''
    Calculate the post-hoc test by Bonferroni correction for a variable with weight values.

    Parameters
    ----------
    df : pd.DataFrame
        The source of dataframe with the variable to be analyzed
    dv : str
        The variable to be analyzed
    groupby : str
        The grouping variable
    w : str, optional
        The weighting variable, by default None, means weight=1
    alpha : float, optional
        The significance level, by default 0.05
    to_clipboard : bool, optional
        Whether to copy the result to the clipboard, by default False

    Returns
    -------
    pandas.DataFrame    
        The post-hoc test result
    '''
    def __new__(cls, df, dv:str, groupby:str, w=None, alpha=0.05, to_clipboard=False):
        # Bonferroni
        result = pd.DataFrame(columns=['group1', 'group2', 'mean_diff', 'Std. Error', 'p-value', 'Sig'])
    
        if w:
            group = sorted(df[groupby].unique())
            group = [df.loc[df[groupby]==g] for g in group]

            adjalpha = alpha / math.comb(len(group), 2)

            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    g1 = group[i]
                    g2 = group[j]
                    g1w = group[i][w]
                    g2w = group[j][w]
                    g1label = group[i][groupby].iloc[0]
                    g2label = group[j][groupby].iloc[0]

                    mean_diff = weighted_AnS(g1, dv, w)[0] - weighted_AnS(g2, dv, w)[0]
                    se = (((weighted_AnS(g1, dv, w)[1]**2)/g1[w].sum()) + ((weighted_AnS(g2, dv, w)[1]**2)/g2[w].sum())) ** 0.5

                    t_stat, t_p_value, ddof = weight_ttest(g1[dv], g2[dv], weights=(g1w, g2w))

                    if t_p_value >= adjalpha:
                        sig = 'False'
                    else:
                        sig = 'True'
                    
                    t_result = pd.DataFrame({'group1': [g1label],
                                            'group2': [g2label],
                                            'mean_diff': [rounding(mean_diff, 3)],
                                            'Std. Error': [rounding(se, 3)],
                                            'p-value': [rounding(t_p_value, 3)],
                                            'Sig': [sig]})
                    result = pd.concat([result, t_result], ignore_index=True)
        else:
            group = sorted(df[groupby].unique())
            group = [df.loc[df[groupby]==g] for g in group]

            adjalpha = alpha / math.comb(len(group), 2)

            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    g1 = group[i]
                    g2 = group[j]
                    g1w = group[i][w]
                    g2w = group[j][w]
                    g1label = group[i][groupby].iloc[0]
                    g2label = group[j][groupby].iloc[0]

                    mean_diff = g1[dv].mean() - g2[dv].mean()
                    se = ((g1[dv].var()/len(g1)) + (g1[dv].var()/len(g1))) ** 0.5

                    t_stat, t_p_value = stats.ttest_ind(g1[dv], g2[dv])

                    if t_p_value >= adjalpha:
                        sig = 'False'
                    else:
                        sig = 'True'
                    
                    t_result = pd.DataFrame({'group1': [g1label],
                                            'group2': [g2label],
                                            'mean_diff': [rounding(mean_diff, 3)],
                                            'Std. Error': [rounding(se, 3)],
                                            'p-value': [rounding(t_p_value, 3)],
                                            'Sig': [sig]})
                    result = pd.concat([result, t_result], ignore_index=True)
        
        if to_clipboard:
            result.to_clipboard()
            return result
        else:
            return result
   
if __name__ == '__main__':
    print(rounding(1234.56789, 2))

    df = pd.DataFrame({'v':[1,2,3,4,5], 'x':[0, 0, 0, 1, 1], 'w':[2,2,2,2,3]})
    print('---freq---')
    print(freq(df, 'v', 'w', ['a', 'b', 'c', 'd', 'e'], to_clipboard=True))
    print('---cross---')
    print(cross(df, 'x', 'v', 'w'))
    print('---weighted_AnS---')
    print(weighted_AnS(df, 'v', 'w'))
    print('---t_test---')
    print(t_test(df, 'v', (df['x']==0), (df['x']==1), w='w', to_clipboard=True))
    print('---one_way_anova---')
    print(one_way_anova(df, 'v', 'x', w='w', to_clipboard=True))
    print('---post_hoc---')
    print(post_hoc(df, 'v', 'x', w='w', to_clipboard=True))
    