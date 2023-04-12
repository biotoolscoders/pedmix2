# Author: Yiming Zhang
import numpy as np
import time
from numba import jit
from UseNumba import *
from UseTools import *

# Calculate the probability of no switchs
@jit(nopython=True)
def Calc_prob_no_switch(p_file, r):
    Prob_Noswitch = np.zeros((len(p_file)))
    for i in range(len(p_file)):
        start_p = float(p_file[i][0])
        end_p = float(p_file[i][-1])
        Prob_Noswitch[i] = (-r*(end_p-start_p))
    return Prob_Noswitch

# Generate the Recombination Probabilities Matrix
@jit(nopython=True)
def Calc_prob_within_segments(Prob_noswitch_bp, length):
    Prob_Noswitch_happen = np.zeros((length,length), dtype=np.float64)
    Prob_Switch_happen = np.zeros((length,length), dtype=np.float64)
    for i in range(length):
        prob_noswitch = 0.0
        prob_switch = np.log(1.0 - np.exp(Prob_noswitch_bp[i]))
        Prob_Noswitch_happen[i][0] = prob_noswitch
        Prob_Switch_happen[i][0] = prob_switch
        for j in range(i+1, length):
            prob_noswitch += Prob_noswitch_bp[j-1]
            prob_switch = prob_noswitch + np.log(1.0 - np.exp(Prob_noswitch_bp[j]))
            Prob_Noswitch_happen[i][j-i] = prob_noswitch
            Prob_Switch_happen[i][j-i] = prob_switch           
    return Prob_Noswitch_happen, Prob_Switch_happen

# Preprocessing
def PreProcess(ref, geno, positionfile, Num_Blocks, recombination_rate, num_refs):
    
    # Read Allele Frequency
    ref_hap_0 = Readfile(ref)
    Num_SNP_per_Segments = len(ref_hap_0[0])//Num_Blocks
    ref_hap = np.zeros((num_refs, Num_Blocks, Num_SNP_per_Segments))
    for x in range(num_refs):
        for i in range(Num_Blocks):
            for j in range(Num_SNP_per_Segments):
                ref_hap[x][i][j] = float(ref_hap_0[x][i*Num_SNP_per_Segments+j])    

    # Haplotypes     
    Haplotype = Readfile(geno)
        
    Haplotype_f = np.zeros((Num_Blocks, Num_SNP_per_Segments))
    for i in range(Num_Blocks):
        for j in range(Num_SNP_per_Segments):
            Haplotype_f[i][j] = float(Haplotype[0][i*Num_SNP_per_Segments+j][0])           
   
    Haplotype_m = np.zeros((Num_Blocks, Num_SNP_per_Segments))
    for i in range(Num_Blocks):
        for j in range(Num_SNP_per_Segments):
            Haplotype_m[i][j] = float(Haplotype[1][i*Num_SNP_per_Segments+j][0])       

    # Positions
    position_0 = Readfile(positionfile)[0]
    
    position = np.zeros((Num_Blocks, Num_SNP_per_Segments+1))

    for i in range(Num_Blocks):
        for j in range(Num_SNP_per_Segments):
            position[i][j] = float(position_0[i*Num_SNP_per_Segments+j])
        if i == Num_Blocks - 1:    
            position[i][-1] = float(position_0[(i+1)*Num_SNP_per_Segments-1])
        else:
            position[i][-1] = float(position_0[(i+1)*Num_SNP_per_Segments])
    
    Prob_Noswitch = Calc_prob_no_switch(position, recombination_rate)
    Prob_No_Switch_Happen, Prob_Switch_Happen = Calc_prob_within_segments(Prob_Noswitch, len(position))

    time_start = time.time()
    
    # Calculate all possible reference probabilities
    
    # Probability Matrix for founders:
    # [[B01 B02 B03 B04]
    #  [B12 B13 B14  0]
    #  [B23 B24  0   0]
    #  [B34  0   0   0]]
    
    Prob_Ref_f = np.zeros((num_refs, Num_Blocks, Num_Blocks))
    for x in range(num_refs):
        for i in range(Num_Blocks):
            Prob_Ref_0 = 0.0
            for j in range(Num_SNP_per_Segments):
                if Haplotype_f[i][j] == 0:
                    Prob_Ref_0 += np.log(ref_hap[x][i][j])
                elif Haplotype_f[i][j] == 1:
                    Prob_Ref_0 += np.log(1.0-ref_hap[x][i][j])
            Prob_Ref_f[x][i][0] = Prob_Ref_0     
        for i in range(Num_Blocks):
            Prob_Ref_1 = Prob_Ref_f[x][i][0]
            for j in range(i+1, Num_Blocks):
                Prob_Ref_1 += Prob_Ref_f[x][j][0]
                Prob_Ref_f[x][i][j-i] = Prob_Ref_1        
                
                
    Prob_Ref_m = np.zeros((num_refs, Num_Blocks, Num_Blocks))
    for x in range(num_refs):
        for i in range(Num_Blocks):
            Prob_Ref_0 = 0.0
            for j in range(Num_SNP_per_Segments):
                if Haplotype_m[i][j] == 0:
                    Prob_Ref_0 += np.log(ref_hap[x][i][j])
                elif Haplotype_m[i][j] == 1:
                    Prob_Ref_0 += np.log(1.0-ref_hap[x][i][j])
            Prob_Ref_m[x][i][0] = Prob_Ref_0     
        for i in range(Num_Blocks):
            Prob_Ref_1 = Prob_Ref_m[x][i][0]
            for j in range(i+1, Num_Blocks):
                Prob_Ref_1 += Prob_Ref_m[x][j][0]
                Prob_Ref_m[x][i][j-i] = Prob_Ref_1                  

    time_end = time.time()
    #print('AI time cost:', time_end-time_start, 's')
    return Prob_No_Switch_Happen, Prob_Switch_Happen, Prob_Ref_f, Prob_Ref_m