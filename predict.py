#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 20:59:22 2018

@author: esmeraldaarriaga
"""
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly
import glob 
import os

path = r'/Users/esmeraldaarriaga/Documents/secciondestino'                     #use your path

allFiles = sorted(glob.glob(path + "/*.csv"), key=os.path.abspath)
temp = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, encoding="latin-1")
    list_.append(df)
temp = pd.concat(list_)

#%%
partida = temp.drop('Unnamed: 7', 1)
pardestino=pd.DataFrame(partida.drop('Unnamed: 3',1))
pardestino.rename(columns={'Dólares':'Dolares','Unnamed: 1':'Seccion','Unnamed: 2':'Partida','Unnamed: 4':'Pais','Unnamed: 5':'Fecha','Dólares.1':'Dinero'}, inplace=True)

groupbypais=(pardestino.groupby('Pais').sum()).sum(axis=1)
groupbypais=groupbypais.reset_index()
groupbypais=groupbypais.sort_values([0],ascending=[False])
groupbypais=groupbypais.reset_index(drop=True)

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
for s in np.arange(0,10,1):
   if s <= 10:
       ind=(df_mod['Pais']==max_dlls_pais.loc[s,'Pais'])
       df_mod.loc[ind,'Pais_reduccion']=max_dlls_pais.loc[s,'Pais']
        
#indx=df_mod['Pais_reduccion']!='Otros'
#df_mod=df_mod.loc[indx,:]

scpp_rf = df_mod.groupby(['Dolares','Seccion','Partida','Pais_reduccion','Fecha'], as_index=False).sum()
df_mod=scpp_rf
df_mod.rename(columns={'Pais_reduccion':'Pais'}, inplace=True)
#%%
df_mod['Año'], df_mod['Mes'] = df_mod['Fecha'].str.split('/', 1).str

def replace_text(x,to_replace,replacement):
    try:
        x=x.replace(to_replace,replacement)
    except:
        pass
    return x

df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Mar','Marzo'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Marzozo','Marzo'))

df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Enero','1'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Febrero','2'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Marzo','3'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Abril','4'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Mayo','5'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Junio','6'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Julio','7'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Agosto','8'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Septiembre','9'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Octubre','10'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Noviembre','11'))
df_mod.Mes = df_mod.Mes.apply(replace_text,args=('Diciembre','12'))
 
#yrt=df_mod.Año + "/"+df_mod.Mes
#df_mod['Date'] = yrt.apply(lambda row: (datetime.strptime(row,'%Y/%m')).strftime('%Y/%m'))

#scpp2 = df_mod.groupby(['Pais','Date'], as_index=False).sum()
#scpp2=scpp2.sort_values('Date')
#%%
red=df_mod.loc[((df_mod['Año']>='2016') & (df_mod['Año']<='2017')),:]
a2018=df_mod.loc[(df_mod['Año']=='2017'),:]
a2018['Año']='2018'
red=pd.concat([red,a2018])
ind=red['Año']=='2018'
red['Dinero'].loc[ind,]=0
nomas=red.loc[ind,]
d_8=nomas.set_index(['Dolares','Seccion','Partida','Pais','Año','Mes'])['Dinero'].to_dict()
d=red.set_index(['Dolares','Seccion','Partida','Pais','Año','Mes'])['Dinero'].to_dict()


L=len(d.values())
d_1=np.zeros((len(d.values()),24))
for segment in np.arange(0,len(d.values())-1,1):   
    a=list(d.keys())[segment]
    g=list(a)
    for i in np.arange(0,24,1):
        if g[4]=='2016':
            if g[5]=='1':
                d_1[segment][i:]='NaN'
                break
            else:
                g[5]=str(int(g[5])-1)
        else:
            if g[5]=='1':
                g[4]=str(int(g[4])-1)
                g[5]='12'
            else:
                g[5]=str(int(g[5])-1)  
        t=tuple(g)
        d_1[segment][i]=d[t]
    print(segment)

t_rezagos=pd.DataFrame(d_1)
#%%
#ref=(red.iloc[:,[0,1,2,3,6,7]]).reset_index(drop=True)
#ref=ref.loc[(ref['Año']=='2018'),:]
#ref.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes']
#ref.to_csv('referencia.csv')

#X.to_csv('prueba.csv')
ene_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,0:12]],1)
ene_valid8=ene_valid8.loc[(ene_valid8['Año']=='2018'),:]
ene_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-1','t-2','t-3','t-4','t-5','t-6','t-7','t-8','t-9','t-10','t-11','t-12']
ene_valid8.iloc[:,0:4]=ene_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
ene_valid8.to_csv('ene_valid8.csv')

feb_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,1:13]],1)
feb_valid8=feb_valid8.loc[(feb_valid8['Año']=='2017'),:]
feb_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-2','t-3','t-4','t-5','t-6','t-7','t-8','t-9','t-10','t-11','t-12','t-13']
feb_valid8.iloc[:,0:4]=feb_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
feb_valid8.to_csv('feb_valid8.csv')

mar_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,2:14]],1)
mar_valid8=mar_valid8.loc[(mar_valid8['Año']=='2017'),:]
mar_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-3','t-4','t-5','t-6','t-7','t-8','t-9','t-10','t-11','t-12','t-13','t-14']
mar_valid8.iloc[:,0:4]=mar_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
mar_valid8.to_csv('mar_valid8.csv')

abril_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,3:15]],1)
abril_valid8=abril_valid8.loc[(abril_valid8['Año']=='2017'),:]
abril_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-4','t-5','t-6','t-7','t-8','t-9','t-10','t-11','t-12','t-13','t-14','t-15']
abril_valid8.iloc[:,0:4]=abril_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
abril_valid8.to_csv('abril_valid8.csv')

mayo_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,4:16]],1)
mayo_valid8=mayo_valid8.loc[(mayo_valid8['Año']=='2017'),:]
mayo_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-5','t-6','t-7','t-8','t-9','t-10','t-11','t-12','t-13','t-14','t-15','t-16']
mayo_valid8.iloc[:,0:4]=mayo_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
mayo_valid8.to_csv('mayo_valid8.csv')

junio_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,5:17]],1)
junio_valid8=junio_valid8.loc[(junio_valid8['Año']=='2017'),:]
junio_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-6','t-7','t-8','t-9','t-10','t-11','t-12','t-13','t-14','t-15','t-16','t-17']
junio_valid8.iloc[:,0:4]=junio_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
junio_valid8.to_csv('junio_valid8.csv')

julio_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,6:18]],1)
julio_valid8=julio_valid8.loc[(julio_valid8['Año']=='2017'),:]
julio_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-7','t-8','t-9','t-10','t-11','t-12','t-13','t-14','t-15','t-16','t-17','t-18']
julio_valid8.iloc[:,0:4]=julio_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
julio_valid8.to_csv('julio_valid8.csv')

agosto_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,7:19]],1)
agosto_valid8=agosto_valid8.loc[(agosto_valid8['Año']=='2017'),:]
agosto_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-8','t-9','t-10','t-11','t-12','t-13','t-14','t-15','t-16','t-17','t-18','t-19']
agosto_valid8.iloc[:,0:4]=agosto_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
agosto_valid8.to_csv('agosto_valid8.csv')

sep_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,8:20]],1)
sep_valid8=sep_valid8.loc[(sep_valid8['Año']=='2017'),:]
sep_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-9','t-10','t-11','t-12','t-13','t-14','t-15','t-16','t-17','t-18','t-19','t-20']
sep_valid8.iloc[:,0:4]=sep_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
sep_valid8.to_csv('sep_valid8.csv')

octu_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,9:21]],1)
octu_valid8=octu_valid8.loc[(octu_valid8['Año']=='2017'),:]
octu_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-10','t-11','t-12','t-13','t-14','t-15','t-16','t-17','t-18','t-19','t-20','t-21']
octu_valid8.iloc[:,0:4]=octu_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
octu_valid8.to_csv('octu_valid8.csv')

nov_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,10:22]],1)
nov_valid8=nov_valid8.loc[(nov_valid8['Año']=='2017'),:]
nov_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-11','t-12','t-13','t-14','t-15','t-16','t-17','t-18','t-19','t-20','t-21','t-22']
nov_valid8.iloc[:,0:4]=nov_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
nov_valid8.to_csv('nov_valid8.csv')

dic_valid8=pd.concat([(red.iloc[:,[0,1,2,3,6,7,5]]).reset_index(drop=True),t_rezagos.iloc[:,11:23]],1)
dic_valid8=dic_valid8.loc[(dic_valid8['Año']=='2017'),:]
dic_valid8.columns=['Seccion','Capitulo','Partida','Pais','Año','Mes','Dinero','t-12','t-13','t-14','t-15','t-16','t-17','t-18','t-19','t-20','t-21','t-22','t-23']
dic_valid8.iloc[:,0:4]=dic_valid8.iloc[:,0:4].apply(lambda x: pd.factorize(x)[0])
dic_valid8.to_csv('dic_valid8.csv')
#%%
#enero=pd.concat([((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='12')),:]).iloc[:,0:6].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='11')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='10')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='9')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='8')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='7')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='6')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='5')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='4')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='3')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='2')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2017') & (scpp_rf['Mes']=='1')),:]).iloc[:,5].reset_index(drop=True)),((scpp_rf.loc[((scpp_rf['Año']=='2016') & (scpp_rf['Mes']=='12')),:]).iloc[:,5].reset_index(drop=True))],axis=1)
#enero.columns = ['Seccion', 'Capitulo','Partida','Pais','Fecha','Diciembre','Noviembre','Octubre','Septiembre','Agosto','Julio','Junio','Mayo','Abril','Marzo','Febrero','Enero','Diciembre']
#enero.iloc[:,0]=pd.factorize(enero.iloc[:,0])[0]
#enero.iloc[:,1]=pd.factorize(enero.iloc[:,1])[0]
#enero.iloc[:,2]=pd.factorize(enero.iloc[:,2])[0]
#enero.iloc[:,3]=pd.factorize(enero.iloc[:,3])[0]
