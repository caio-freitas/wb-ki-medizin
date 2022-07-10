import biosppy as bp
import numpy as np 
import matplotlib.pyplot as plt
from biosppy.signals import ecg 


def filter_h_a(ecg_vector):
    """
    ecg_vectr = introduce np.array with the vector 
    
    return: returns the vector with transformations if they increase the number of detected peaks.
    
    """
    _,_,  rpeaks, _, _, _, heart_rate1 =ecg.ecg(ecg_vector, 300, show = False)
    a = np.size(rpeaks)*300/np.size(ecg_vector)
    if a >0.7:
        return ecg_vector
    
    else:
    
        _,_,  before_rpeaks, _, _, _, heart_rate1 =ecg.ecg(ecg_vector, 300, show = False)
            
        if heart_rate1.size == 0:
            return ecg_vector
            
        else:
            _, cortes, _= clean_function(ecg_vector)
            ecg_señal = ecg_vector

            cortes = np.insert(cortes, 0, 0)
            cortes = np.append(cortes,np.size(ecg_vector)-1)

            aux = np.argmax(np.diff(cortes))
        
            comienzo = np.linspace(0, cortes[aux],5000, dtype = int)
            comienzo = np.unique(comienzo) 

            final = np.linspace(cortes[aux+1], np.size(ecg_vector)-1, 5000, dtype = int)
            final = np.unique(final)

            procesado = np.delete(ecg_señal, final)
            procesado = np.delete(procesado, comienzo)
            
            _,_,  after_rpeaks, _, _, _, _ =ecg.ecg(procesado, 300, show = False)
            
            if np.size(before_rpeaks)< np.size(after_rpeaks):
                
                _,_,  after_rpeaks, _, _, _, _ =ecg.ecg(ecg_vector, 300, show = False)
                _,_,  after_rpeaks, _, _, _, _ =ecg.ecg(procesado, 300, show = False)
                result = procesado
            else:
                result = ecg_vector
    return result 







def clean_function(vector_ecg):
    """
    Imputs : 
            siganl :ECG -> array
    
    Outputs: 
            
            signal: ECG transform -> array
            proces: text with what kind of transformation{ 'can not be read by biosppy', 
                                                          'do not need process' ,'it is process by the function' }
             
    Explanation of the function: 
    
    In making this function, it has been taken into account that it is better to modify the essential to try not to introduce more noise in the data.
    In making this function, it has been taken into account that it is better to modify the essential to try not to introduce more noise in the data.
    The function shall first process with the bioppsy library the electrocardiogram. 

    First. If the bioppsy library cannot process the electrocardiogram, the function will return the same electrocardiogram.
    Secondly. If when processing the function the bioppsy library considers that the EKG has low noise it will not process the function.
    low noise: low heart rate (<100), small heart rate deviation (<10) and processed signal length not less than 90%.
    Third. If the above conditions are not met, the function shall process the electrocardiogram.

    We pre-calculate the number(sigma) of standard deviations that introduce no more than approximately 10% change in ECG length.

    We cut all those peaks and points close to the peaks that exceed the sigma number of standard deviations of the ECG.
    """
    _,filtered, _, _, _, _, heart_rate1=ecg.ecg(vector_ecg, 300, show= False)
    
    
    if heart_rate1.size== 0 or filtered.size ==0:
        signal = vector_ecg
        process= 'can not be read by biosppy'
        cortes = np.array([0, np.size(vector_ecg)-1])
        
    
 #    elif (int(np.mean(heart_rate1))<105 and int(np.std(heart_rate1))<10) and 0.9<(np.size(filtered)/np.size(vector_ecg)):
 #       
 #       signal = vector_ecg
 #       process = 'do not need process'
    
    
    
    else: 
        sigma= 1
        cortes, centradocero= calcular_sigma(vector_ecg, sigma)
        
        while np.size(cortes)/np.size(centradocero) >= 0.15:
            sigma = sigma+ 0.25
            cortes, centradocero =calcular_sigma(vector_ecg, sigma)
        
        signal, cortes  =  clean_big_mistake(vector_ecg, sigma)
        process = 'it is process by the function'
    
    return signal, cortes, process




def calcular_sigma_1(vector_ecg, sigma):
    """
     
    Imputs : 
            vector_ecg :ECG -> array
            sigma -> float
    Outputs: 
            
            cortes : index of points to cut in ECG_preprocessed -> array
            centradocero: ECG_preprocesed ->array
    
    """
    #calculamos la media del vector, lo centramos en cero y calcualmos la desviación estandar.
    mean = np.mean(vector_ecg)
    mean = np.full(np.size(mean),mean)
    centradocero = vector_ecg - mean
    std = np.std(vector_ecg)


    #inciamos  el vector en un punto cercano al cero y acabamos el vector en un punto cercano al cero


    indices_corte_cero= np.where(((centradocero >-20) & (centradocero<20)))
    centradocero =np.delete(centradocero, np.linspace(0,indices_corte_cero[0][0],indices_corte_cero[0][0]+1,dtype=int ))
    indices_corte_cero= np.where(((centradocero >-20) & (centradocero<20)))
    centradocero =np.delete(centradocero, np.linspace(indices_corte_cero[0][np.size(indices_corte_cero)-1]-1,-1+np.size(centradocero),np.size(centradocero)-indices_corte_cero[0][np.size(indices_corte_cero)-1]+1  ,dtype=int ) )
    
    
    std_vector = np.full(np.size(centradocero),std)

    #hacemos los cálculos donde cortan las las bandas con el electrocariograma y sacamos los indices donde esán esos cortes

    cortespositivos = centradocero - sigma*std_vector
    cortesnegativos = centradocero + sigma*std_vector

    indice_de_corte = np.array([], int)

    for i in range(np.size(cortespositivos)-1):
        if cortespositivos[i]*cortespositivos[i+1]<0:
            indice_de_corte = np.append(indice_de_corte,i)
        

    for i in range(np.size(cortesnegativos)-1):
        if cortesnegativos[i]*cortesnegativos[i+1]<0:
            indice_de_corte = np.append(indice_de_corte,i)

    #una vez que tenemos los indices donde cruzan las bandas con el electrocardiograma definimos los puntos donde debemos cortar


    cortes = np.array([], int)
    for i in range(int(np.size(indice_de_corte)/2)):
        
        cortes = np.append(cortes,np.linspace(int(indice_de_corte[2*i]-60), int(indice_de_corte[2*i+1]+120), 5000,dtype = int ) )
    
    cortes = np.unique(cortes)
    return(cortes, centradocero)

#vemos como es el electrocardiogrma sin ninguna modificación y vemos como es cuando lo pasamos
#por la función bioppsy sin haber hecho nada. Tambien calculamos el heart_rate.
def calcular_sigma(vector_ecg, sigma):
    
    
    #fig, ax = plt.subplots(figsize=(19,3))
    #ax.plot(np.linspace(1,np.size(vector_ecg), np.size(vector_ecg)),vector_ecg )
    #plt.set_tittle('ECG without transformation')
    
 


    #calculamos la media del vector, lo centramos en cero y calcualmos la desviación estandar.
    mean = np.mean(vector_ecg)
    mean = np.full(np.size(mean),mean)
    centradocero = vector_ecg - mean
    std = np.std(vector_ecg)


    #inciamos  el vector en un punto cercano al cero y acabamos el vector en un punto cercano al cero


    indices_corte_cero= np.where(((centradocero >-20) & (centradocero<20)))
    centradocero =np.delete(centradocero, np.linspace(0,indices_corte_cero[0][0],indices_corte_cero[0][0]+1,dtype=int ))
    indices_corte_cero= np.where(((centradocero >-20) & (centradocero<20)))
    centradocero =np.delete(centradocero, np.linspace(indices_corte_cero[0][np.size(indices_corte_cero)-1]-1,-1+np.size(centradocero),np.size(centradocero)-indices_corte_cero[0][np.size(indices_corte_cero)-1]+1  ,dtype=int ) )
    


    std_vector = np.full(np.size(centradocero),std)

    #hacemos los gráficos de bandas donde debe estar situada la mayor parte del electrocardiograma
    #fig1, ax1 = plt.subplots(figsize=(19,3))
    #plt.set_tittle('Centrate in zero, start in cero and cut with bands')
    #ax1.plot(np.linspace(1,np.size(centradocero), np.size(centradocero)), centradocero)
    #ax1.plot(np.linspace(1,np.size(centradocero), np.size(centradocero)), sigma*std_vector)
    #ax1.plot(np.linspace(1,np.size(centradocero), np.size(centradocero)), -sigma*std_vector)

    #hacemos los cálculos donde cortan las las bandas con el electrocariograma y sacamos los indices donde esán esos cortes

    cortespositivos = centradocero - sigma*std_vector
    cortesnegativos = centradocero + sigma*std_vector

    indice_de_corte = np.array([], int)

    for i in range(np.size(cortespositivos)-1):
        if cortespositivos[i]*cortespositivos[i+1]<0:
            indice_de_corte = np.append(indice_de_corte,i)
        

    for i in range(np.size(cortesnegativos)-1):
        if cortesnegativos[i]*cortesnegativos[i+1]<0:
            indice_de_corte = np.append(indice_de_corte,i)

    #una vez que tenemos los indices donde cruzan las bandas con el electrocardiograma definimos los puntos donde debemos cortar


    cortes = np.array([], int)
    for i in range(int(np.size(indice_de_corte)/2)):
        
        cortes = np.append(cortes,np.linspace(int(indice_de_corte[2*i]-60), int(indice_de_corte[2*i+1]+120), 5000,dtype = int ) )
    
    cortes = np.unique(cortes)
    return(cortes, centradocero)


def clean_big_mistake(vector_ecg, sigma):
    
    """
     
    Imputs : 
            vector_ecg :ECG -> array
            sigma -> float
    Outputs: 
            
    
            ECG_procesed: ECG_preprocesed ->array
    
    """
    
    
    
    
    #calculamos la media del vector, lo centramos en cero y calcualmos la desviación estandar.
    mean = np.mean(vector_ecg)
    mean = np.full(np.size(mean),mean)
    centradocero = vector_ecg - mean
    std = np.std(vector_ecg)
    
    
    #inciamos  el vector en un punto cercano al cero y acabamos el vector en un punto cercano al cero
    
    
    indices_corte_cero= np.where(((centradocero >-20) & (centradocero<20)))
    centradocero =np.delete(centradocero, np.linspace(0,indices_corte_cero[0][0],indices_corte_cero[0][0]+1,dtype=int ))
    indices_corte_cero= np.where(((centradocero >-20) & (centradocero<20)))
    centradocero =np.delete(centradocero, np.linspace(indices_corte_cero[0][np.size(indices_corte_cero)-1]-1,-1+np.size(centradocero),np.size(centradocero)-indices_corte_cero[0][np.size(indices_corte_cero)-1]+1  ,dtype=int ) )
    
    std_vector = np.full(np.size(centradocero),std)
    
    
    #hacemos los cálculos donde cortan las las bandas con el electrocariograma y sacamos los indices donde esán esos cortes
    
    cortespositivos = centradocero - sigma*std_vector
    cortesnegativos = centradocero + sigma*std_vector
    a = np.zeros(np.size(cortespositivos))
    indice_de_corte = np.array([], int)

    for i in range(np.size(cortespositivos)-1):
        if cortespositivos[i]*cortespositivos[i+1]<0:
            indice_de_corte = np.append(indice_de_corte,i)
        

    for i in range(np.size(cortesnegativos)-1):
        if cortesnegativos[i]*cortesnegativos[i+1]<0:
            indice_de_corte = np.append(indice_de_corte,i)

    #una vez que tenemos los indices donde cruzan las bandas con el electrocardiograma definimos los puntos donde debemos cortar


    cortes = np.array([], int)
    for i in range(int(np.size(indice_de_corte)/2)):
        
        cortes = np.append(cortes,np.linspace(int(indice_de_corte[2*i]-60), int(indice_de_corte[2*i+1]+120),5000,dtype = int )) 
    cortes = np.unique(cortes)
    # realizamos los cortes
    
    if cortes.size==0:
        cortes = np.size([0, np.size(centradocero)-1])
        ECG_procesed = centradocero
       
    # vemos si hay problemas debido a que tenemos que realizar cortes cuando el vector ya ha acabado
    elif cortes[int(np.size(cortes)-1)]>np.size(centradocero):
        p = np.append(centradocero,np.zeros(cortes[np.size(cortes)-1]+1-np.size(centradocero)))
        ECG_procesed = np.array([])
        ECG_procesed = np.delete(p, cortes)
       
    
    # vemos si hay problemas debido a que tenemos que realizar cortes cuando el vector no ha empezado
    elif cortes[0]<0:
        cortesito = np.where(cortes == 0)
        cortes =np.delete(cortes, np.linspace(0,int(cortesito[0][0]),int(cortesito[0][0]+1),dtype=int ))
        ECG_procesed = np.array([])
        ECG_procesed = np.delete(centradocero, cortes)
      
        
        
    #realizamos los cortes  si no hay ninguno de los problemas anteriores.
    else:
        ECG_procesed = np.array([])
        cortes = np.delete(cortes, np.size(cortes)-1) 
        ECG_procesed = np.delete(centradocero, cortes)
       

    return ECG_procesed, cortes
