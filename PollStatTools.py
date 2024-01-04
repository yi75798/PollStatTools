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
# import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.weightstats import ttest_ind as weight_ttest
# from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import math

## Rounding
def rounding(num, decimal=0):
    str_deci = 1
    if decimal != 0:
        str_deci /= (10 ** decimal)

        str_deci = str(str_deci)
        result = Decimal(str(num)).quantize(Decimal(str_deci), rounding=ROUND_HALF_UP)
        result = float(result)
        return result
    else:
        str_deci /= (10 ** 1)
        str_deci = str(str_deci)
        result = Decimal(str(num)).quantize(Decimal(str_deci), rounding=ROUND_HALF_UP)

        result = Decimal(str(result)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
        result = int(result)
        return result

### Frequency
def freq(data, var: str, w=None, label=None, to_clipboard=False):
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

# freq(df.loc[~df['v1'].isin([90, 97, 98])], 'v1', w='weight',
#     label=['YouTube', '臉書', '電視', '報紙', 'LINE', '其他', 'Goolge新聞', 'Instagram', 'PTT', 'Yahoo奇摩新聞', '廣播'],
#     to_clipboard=True)

# freq(df, 'v27', w='weight',
#     label=['DPP', 'KMT', 'TPP', 'NPP', '其他', 'Pan_blue', 'Pan_green', '都沒有', '基進', '不知道', '拒答'],
#     to_clipboard=True)

### cross table
def cross(data, r_var: str, c_var: str ,w=None, percent_by='r', to_clipboard=False):
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

# cross(df, 'v27', 'v28', 'weight', to_clipboard=True)

### t-test
def weighted_AnS(df, var:str, weight: str):
    average = np.average(df[var], weights=df[weight])
    
    variance = np.average((df[var]-average)**2, weights=df[weight])
    return (average, math.sqrt(variance))

def t_test(df, var:str, group1, group2, w=None, group1_label='group1', group2_label='group2', to_clipboard=False):
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

# group1 = (dft['sex'] == 1)
# group2 = (dft['sex'] == 2)

# t_test(dft, 'v10', group1, group2, w='weight', group1_label='Male', group2_label='Female', to_clipboard=True)
# t_test(dft, 'v10', group1, group2)

### ANOVA test
def one_way_anova(df, dv:str, groupby:str, w=None, to_clipboard=False):
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
        dfw = df['weight'].sum() - len(n)
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

# one_way_anova(dft, 'v11', 'v27')
# one_way_anova(dft, 'v11', 'v27', w='weight')

###
def post_hoc(df, dv:str, groupby:str, w=None, alpha=0.05, to_clipboard=False):
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
                se = (((weighted_AnS(g1, dv, w)[1]**2)/g1['weight'].sum()) + ((weighted_AnS(g2, dv, w)[1]**2)/g2['weight'].sum())) ** 0.5

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

# post_hoc(dft, 'v11', 'v27', 'weight')
    
