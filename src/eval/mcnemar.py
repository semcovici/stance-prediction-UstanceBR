# -*- coding: utf-8 -*-
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

def main_mcnemar(
    # tipo,
    pre,
    pos
    ):

    # objetivo: verificar se dois classificadores (aqui chamados pre e pos)
    #           sao significativamente distintos
    # entrada:  dois vetores com os acertos (1) ou erros (0) de cada um dos
    #           dois classificadores a serem comparados
    # saída:    a estatística McNemar e seu indicativo de significância p
   
    # nota: Match não é a coluna com a predição do classificador. É uma coluna com
    # um valor 0 ou 1 indicando se a predição que foi feita é correta ou não

    df = pd.concat([pre,pos],axis=1) # lado
    df.columns = ['pre','pos']

    df['right_right'] = df.apply(lambda x: int(x.pre==1 and x.pos==1),axis=1)
    df['right_wrong'] = df.apply(lambda x: int(x.pre==1 and x.pos==0),axis=1)
    df['wrong_right'] = df.apply(lambda x: int(x.pre==0 and x.pos==1),axis=1)
    df['wrong_wrong'] = df.apply(lambda x: int(x.pre==0 and x.pos==0),axis=1)

    cell_11 = df.right_right.sum()
    cell_12 = df.right_wrong.sum()
    cell_21 = df.wrong_right.sum()
    cell_22 = df.wrong_wrong.sum()
    
    table = [
             [cell_11, cell_12],
             [cell_21, cell_22] 
            ]
                
    if cell_11<25 or cell_12<25 or cell_21<25 or cell_22<25:
        result = mcnemar(table, exact=True)
    else:
        result = mcnemar(table, exact=False, correction=True)   

    # print('\n'+tipo+': ',end='')
    # print('stat=%.3f, p=%.7f' % (result.statistic, result.pvalue),end='')

    significant = False
    for alpha in [0.001, 0.01, 0.05]:
        if result.pvalue < alpha:
            # print(' p<' + str(alpha))
            significant = True
    if not significant:
        pass
        # print(' not significant')
        
    return result.pvalue
            

# # -*- coding: utf-8 -*-
# from Util_posdiag import GetRoot
# import pandas as pd
# from statsmodels.stats.contingency_tables import mcnemar

# def main_mcnemar(mode,tipo):

#     # objetivo: verificar se dois classificadores (aqui chamados pre e pos)
#     #           sao significativamente distintos
#     # entrada:  dois vetores com os acertos (1) ou erros (0) de cada um dos
#     #           dois classificadores a serem comparados
#     # saída:    a estatística McNemar e seu indicativo de significância p
   
#     rootpath  = GetRoot() + 'corpus\\posdiag\\logreg_' + mode + "\\"
#     predir = rootpath + 'predictions\\'
    
#     # nota: Match não é a coluna com a predição do classificador. É uma coluna com
#     # um valor 0 ou 1 indicando se a predição que foi feita é correta ou não
#     pre = pd.read_csv(predir+tipo+"_pre_predictions.csv",sep=';',usecols=['Match'])
#     pos = pd.read_csv(predir+tipo+"_pos_predictions.csv",sep=';',usecols=['Match'])

#     df = pd.concat([pre,pos],axis=1) # lado
#     df.columns = ['pre','pos']

#     df['right_right'] = df.apply(lambda x: int(x.pre==1 and x.pos==1),axis=1)
#     df['right_wrong'] = df.apply(lambda x: int(x.pre==1 and x.pos==0),axis=1)
#     df['wrong_right'] = df.apply(lambda x: int(x.pre==0 and x.pos==1),axis=1)
#     df['wrong_wrong'] = df.apply(lambda x: int(x.pre==0 and x.pos==0),axis=1)

#     cell_11 = df.right_right.sum()
#     cell_12 = df.right_wrong.sum()
#     cell_21 = df.wrong_right.sum()
#     cell_22 = df.wrong_wrong.sum()
    
#     table = [
#              [cell_11, cell_12],
#              [cell_21, cell_22] 
#             ]
                
#     if cell_11<25 or cell_12<25 or cell_21<25 or cell_22<25:
#         result = mcnemar(table, exact=True)
#     else:
#         result = mcnemar(table, exact=False, correction=True)   

#     print('\n'+tipo+': ',end='')
#     print('stat=%.3f, p=%.7f' % (result.statistic, result.pvalue),end='')

#     significant = False
#     for alpha in [0.001, 0.01, 0.05]:
#         if result.pvalue < alpha:
#             print(' p<' + str(alpha))
#             significant = True
#             break
#     if not significant:
#         print(' not significant')
            