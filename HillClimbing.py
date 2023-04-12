# Author: Yiming Zhang
import numpy as np
from numba import jit
from DynamicProgramming import *
from UseNumba import *

# Hill Climbing Inference
@jit(nopython=True)
def Infer(List_founders, Identical, Ref_f, Ref_m, length, PNoS, PS, NodeList, Num_generations, num_Chros, Num_Ref):
    Num_founder = 2 ** (Num_generations - 1)
    List_f = RNF(List_founders[:Num_founder])
    List_m = RNF(List_founders[Num_founder:])
    List = np.concatenate((List_f, List_m))
    founders_Opt = np.ones((2 ** (Num_generations)))

    List_byte = ArraytoString(List)
    
    if List_byte in Identical:
        return None, np.iinfo(np.int32).min, Identical
    else:
        Identical.append(List_byte)
        DPTable_Opts_f = np.zeros((num_Chros, 2, 2**(Num_generations+1)-1, 2, length, length))
        DPTable_Opts_m = np.zeros((num_Chros, 2, 2**(Num_generations+1)-1, 2, length, length))
        Prob_Opts = np.zeros((num_Chros))
        for i in range(num_Chros):
            DPTable_Opts_f[i, 0, :, :, :, :] = Calc_Prob_fast(List_founders[:Num_founder], Ref_f[i], length, PNoS[i], PS[i], NodeList, Num_generations)
            DPTable_Opts_f[i, 1, :, :, :, :] = Calc_Prob_fast(List_founders[:Num_founder], Ref_m[i], length, PNoS[i], PS[i], NodeList, Num_generations)
            DPTable_Opts_m[i, 0, :, :, :, :] = Calc_Prob_fast(List_founders[Num_founder:], Ref_m[i], length, PNoS[i], PS[i], NodeList, Num_generations)
            DPTable_Opts_m[i, 1, :, :, :, :] = Calc_Prob_fast(List_founders[Num_founder:], Ref_f[i], length, PNoS[i], PS[i], NodeList, Num_generations)
            Prob_Opt_ind_f_1 = logA(DPTable_Opts_f[i, 0, -1, 0, 0, -1], DPTable_Opts_f[i, 0, -1, 1, 0, -1])
            Prob_Opt_ind_f_2 = logA(DPTable_Opts_f[i, 1, -1, 0, 0, -1], DPTable_Opts_f[i, 1, -1, 1, 0, -1])
            Prob_Opt_ind_m_1 = logA(DPTable_Opts_m[i, 0, -1, 0, 0, -1], DPTable_Opts_m[i, 0, -1, 1, 0, -1])
            Prob_Opt_ind_m_2 = logA(DPTable_Opts_m[i, 1, -1, 0, 0, -1], DPTable_Opts_m[i, 1, -1, 1, 0, -1])
            Prob_Opt_ind = logA((Prob_Opt_ind_f_1 + Prob_Opt_ind_m_1), (Prob_Opt_ind_f_2 + Prob_Opt_ind_m_2))
            Prob_Opts[i] = Prob_Opt_ind
        Prob_Opt = sum(Prob_Opts)

        founders_Opt = List_founders.copy()

        change = True
        Last_Ratio = np.ones((2 ** (Num_generations)))
        while(change is True):
            change = False
            Last_Ratio[:] = founders_Opt
            current_DPTable_f = DPTable_Opts_f.copy()
            current_DPTable_m = DPTable_Opts_m.copy()
            new_list_o = np.ones((2 ** (Num_generations)))
            old_list_o = np.ones((2 ** (Num_generations)))

            for j in range(len(Last_Ratio)):
                for w in range(Num_Ref - 1):
                    new_list_o[:] = Last_Ratio
                    old_list_o[:] = Last_Ratio
                    if new_list_o[j] + w >= Num_Ref - 1:
                        new_list_o[j] = new_list_o[j] + w - (Num_Ref - 1)
                    else:    
                        new_list_o[j] = new_list_o[j]+w+1
                    new_list_f = RNF(new_list_o[:Num_founder])
                    new_list_m = RNF(new_list_o[Num_founder:])
                    new_list = np.concatenate((new_list_f, new_list_m))
                    new_list_byte = ArraytoString(new_list)
                    if new_list_byte not in Identical:
                        Prob_news = np.zeros((num_Chros))
                        if j < Num_founder:
                            for i in range(num_Chros):            
                                Calc_Prob_Hill_Clim(j, new_list_o[j], current_DPTable_f[i,0], Ref_f[i], length, PNoS[i], PS[i], NodeList)
                                Calc_Prob_Hill_Clim(j, new_list_o[j], current_DPTable_f[i,1], Ref_m[i], length, PNoS[i], PS[i], NodeList)                         
                                Prob_new_ind_f_1 = logA(current_DPTable_f[i, 0, -1, 0, 0, -1], current_DPTable_f[i, 0, -1, 1, 0, -1])
                                Prob_new_ind_f_2 = logA(current_DPTable_f[i, 1, -1, 0, 0, -1], current_DPTable_f[i, 1, -1, 1, 0, -1])
                                Prob_new_ind_m_1 = logA(current_DPTable_m[i, 0, -1, 0, 0, -1], current_DPTable_m[i, 0, -1, 1, 0, -1])
                                Prob_new_ind_m_2 = logA(current_DPTable_m[i, 1, -1, 0, 0, -1], current_DPTable_m[i, 1, -1, 1, 0, -1])
                                Prob_new_ind = logA((Prob_new_ind_f_1 + Prob_new_ind_m_1), (Prob_new_ind_f_2 + Prob_new_ind_m_2))
                                Prob_news[i] = Prob_new_ind
                            Prob_new = sum(Prob_news)
                            Identical.append(new_list_byte)
                            if Prob_new > Prob_Opt:
                                Prob_Opt = Prob_new
                                founders_Opt[:] = new_list_o
                                DPTable_Opts_f = current_DPTable_f.copy()
                                DPTable_Opts_m = current_DPTable_m.copy()
                                change = True
                            for i in range(num_Chros):                    
                                Calc_Prob_Hill_Clim(j, old_list_o[j], current_DPTable_f[i,0], Ref_f[i], length, PNoS[i], PS[i], NodeList)
                                Calc_Prob_Hill_Clim(j, old_list_o[j], current_DPTable_f[i,1], Ref_m[i], length, PNoS[i], PS[i], NodeList)

                        elif j >= Num_founder:
                            for i in range(num_Chros):          
                                Calc_Prob_Hill_Clim(j-Num_founder, new_list_o[j], current_DPTable_m[i,0], Ref_m[i], length, PNoS[i], PS[i], NodeList)
                                Calc_Prob_Hill_Clim(j-Num_founder, new_list_o[j], current_DPTable_m[i,1], Ref_f[i], length, PNoS[i], PS[i], NodeList)       
                                Prob_new_ind_f_1 = logA(current_DPTable_f[i, 0, -1, 0, 0, -1], current_DPTable_f[i, 0, -1, 1, 0, -1])
                                Prob_new_ind_f_2 = logA(current_DPTable_f[i, 1, -1, 0, 0, -1], current_DPTable_f[i, 1, -1, 1, 0, -1])
                                Prob_new_ind_m_1 = logA(current_DPTable_m[i, 0, -1, 0, 0, -1], current_DPTable_m[i, 0, -1, 1, 0, -1])
                                Prob_new_ind_m_2 = logA(current_DPTable_m[i, 1, -1, 0, 0, -1], current_DPTable_m[i, 1, -1, 1, 0, -1])
                                
                                Prob_new_ind = logA((Prob_new_ind_f_1 + Prob_new_ind_m_1), (Prob_new_ind_f_2 + Prob_new_ind_m_2))
                                Prob_news[i] = Prob_new_ind
                            Prob_new = sum(Prob_news)
                            Identical.append(new_list_byte)
                            if Prob_new > Prob_Opt:
                                Prob_Opt = Prob_new
                                founders_Opt[:] = new_list_o
                                DPTable_Opts_m = current_DPTable_m.copy()
                                DPTable_Opts_f = current_DPTable_f.copy()
                                change = True  
                            for i in range(num_Chros):                    
                                Calc_Prob_Hill_Clim(j-Num_founder, old_list_o[j], current_DPTable_m[i,0], Ref_m[i], length, PNoS[i], PS[i], NodeList)
                                Calc_Prob_Hill_Clim(j-Num_founder, old_list_o[j], current_DPTable_m[i,1], Ref_f[i], length, PNoS[i], PS[i], NodeList)   
                    else:
                        continue

        return founders_Opt, Prob_Opt, Identical