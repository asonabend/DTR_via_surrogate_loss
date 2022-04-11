from utils import *
import os, sys
#size,sims_No,method,phi_No = int(sys.argv[2]),  int(sys.argv[6]), str(sys.argv[8]), str(sys.argv[10])
size,sims_No,method = int(sys.argv[2]),  int(sys.argv[6]), str(sys.argv[8])
setting = int(sys.argv[4]) if str(sys.argv[4]).isdigit() else str(sys.argv[4])
if method in ['linear','NN','wavelets','splines','SOWLRBF','SOWLlinear','DQlearning','linQlearning']:
    learningRate = 0.01 if method in ['linear','linQlearning'] and setting != 1 else .001
    #phi_No = int(phi_No) if method in ['linear','NN','wavelets','splines'] else None
    learningRate, epochs = (learningRate,20) if 'SOWL' not in method  else (None,None)
    if method in ['linear','NN','wavelets','splines']:
        sys.stderr.write('\n Running '+'surrogate loss with '+method+' functions\n')
    else:
        sys.stderr.write('\n Running '+method+' benchmark')
    sys.stderr.write('\nn = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')
    _ = simulations(size,setting,sims_No,f_model=method,learningRate = learningRate,epochs =epochs,phi_No=None,CV_K=5)        
else:
    sys.stderr.write('Error: method not valid')