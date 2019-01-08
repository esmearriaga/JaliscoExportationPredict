#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 09:45:37 2018

@author: esmeraldaarriaga
"""

import glob 
import os
import pandas as pd
import numpy as np
path = r'/Users/esmeraldaarriaga/Documents/secciondestino'                     #use your path

allFiles = sorted(glob.glob(path + "/*.csv"), key=os.path.abspath)
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, encoding="latin-1")
    list_.append(df)
frame = pd.concat(list_)

#%% concat con reduccion

partida = frame.drop('Unnamed: 7', 1)

pardestino=pd.DataFrame(partida.drop('Unnamed: 3',1))
pardestino.rename(columns={'Dólares':'Dolares','Unnamed: 1':'Seccion','Unnamed: 2':'Partida','Unnamed: 4':'Pais'}, inplace=True)

groupbypais=(pardestino.groupby('Pais').sum()).sum(axis=1)

#%%

porcentaje = (pd.DataFrame(sorted(groupbypais, reverse=True))/np.sum(groupbypais))*100
porcentaje_acum = np.cumsum(porcentaje)

plt.title('Porcentaje acumulado')
plt.step(np.arange(len(porcentaje_acum)),porcentaje_acum)
plt.show()

#%%
max_dlls_pais = pd.DataFrame(groupbypais.nlargest(25))
max_dlls_pais.reset_index(level=0, inplace=True)

#%%
df_mod = pardestino.copy()

df_mod['Pais_reduccion'] = "Otros"


#%%
for s in np.arange(0,25,1):
   if s <= 25:
       ind=(df_mod['Pais']==max_dlls_pais.loc[s,'Pais'])
       df_mod.loc[ind,'Pais_reduccion']=max_dlls_pais.loc[s,'Pais']
        
indx=df_mod['Pais_reduccion']!='Otros'
df_mod=df_mod.loc[indx,:]