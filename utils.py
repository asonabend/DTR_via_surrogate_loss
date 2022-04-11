# import the required packages
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import math
import random
import time
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import itertools
#  cubic splines:
from patsy import dmatrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
# wavelets:
import pywt
from itertools import combinations

from tqdm import tqdm
import os, sys
# cross-val:
from sklearn.model_selection import KFold
pandas2ri.activate()


##############################
###################################################################################################
################################################################################################################################
#df.head()
# Generate dataset using R:

#path = '~/Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Code/Simulations/'
path = ''
def gen_data(size,setting,sd):
    r = ro.r
    # load the R functions
    r.source(path+"functions.R")
    # generate dataset
    df = r.gen_df(size,setting,sd)
    return df

def BOWL_optim(df,df_test,setting,O1_vars = None,O2_vars = None, surr_fn =  'hinge', kernel = 'linear'):
    if setting  == 'sepsis':
        O1,O2 = O1_vars,O2_vars
    elif setting == 1:
        O1,O2 = ['O1.1', 'O1.2', 'O1.3'],[]
    elif setting == 4:
        O1, O2 = ['O'+str(i) for i in range(1,7)], ['Z.21', 'Z.22']
    elif setting == 5:
        O1, O2 = ['X1'], ['X1','X2']
    elif setting in list(range(10,14)):
        O1, O2 = ['X1','O11','O12'], ['X1','X2','O11','O12','O23']
    else:
        O1, O2 = ['O1.1', 'O1.2', 'O1.3'], ['O2.1', 'O2.2']
    O2_bar = list(set(O1+O2+['A1.f','Y1']))
    r = ro.r
    # load the R functions
    r.source(path+"functions.R")
    # fit BOWL
    #BOWL_optim\(df.train,df.test,O1.vars,O2.bar.vars,surr_fn =  'hinge')#'exp')#
    #df = df.rename(columns={"A1": "A1.f", "A2": "A2.f"})
    #df_test = df_test.rename(columns={"A1": "A1.f", "A2": "A2.f"})
    df[['A1.f','A2.f']] = df[['A1','A2']].astype(int)
    df_test[['A1.f','A2.f']] = df_test[['A1','A2']].astype(int)
    df['Y1'] += abs(min(df['Y1']))
    df['Y2'] += abs(min(df['Y2']))
    start_time = time.clock()
    df_res = r.fit_bowl(df,df_test,O1,O2_bar,surr_fn,kernel)
    end_time = time.clock()
    d1_hat = 2*(np.sign(df_res['d1.hat.bowl.'+surr_fn])  > 0) - 1
    d2_hat = 2*(np.sign(df_res['d2.hat.bowl.'+surr_fn])  > 0) - 1
    compute_time = end_time-start_time
    return(np.stack((d1_hat,d2_hat),axis=1),compute_time)

class net(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(net, self).__init__()
        self.fc1 = nn.Linear(inputSize, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, outputSize)
        self.DO = nn.Dropout(p=0.5)
    def forward(self, x):
        x = F.relu(self.DO(self.fc1(x)))
        x = F.relu(self.DO(self.fc2(x)))
        out = self.fc3(x)
        return out    

class lm(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(lm, self).__init__()
        self.linear_reg = nn.Linear(inputSize,outputSize)
    def forward(self, x):
        out = self.linear_reg(x)
        return out    

def phi(x,phi_No):
    if phi_No == 1:
        return 1+x/torch.sqrt(1+x**2)
    elif phi_No == 2:
        return 1+x/(1+torch.abs(x))
    elif phi_No == 3:
        return 1+(2/math.pi)*torch.atan(math.pi*x/2)
    elif phi_No == 4:
        return 2/(1+torch.exp(-x))
    else:
        return 1+(2/math.pi)*torch.tanh(math.pi*x/2)

def my_loss(f1,f2,f1_hat,f2_hat,A1,A2,Y1,Y2,phi_No,l1_reg,lamb,alpha,cnvx_lss=None):
    loss = -torch.mean((Y1+Y2)*phi(f1_hat*A1,phi_No)*phi(f2_hat*A2,phi_No))
    if l1_reg:    
        regularization_loss = 0
        for param in f1.parameters():
            regularization_loss += alpha * torch.sum(param**2) + (1-alpha) * torch.sum(abs(param))
        for param in f2.parameters():
            regularization_loss += alpha * torch.sum(param**2) + (1-alpha) * torch.sum(abs(param))
        loss += lamb * regularization_loss
    if cnvx_lss is not None:
        x,y = f1_hat*A1,f2_hat*A2         
        if cnvx_lss == 1:
            loss = -torch.mean((Y1+Y2)*(-(x-1)**2-(y-1)**2))
        elif cnvx_lss == 2:
            loss = -torch.mean((Y1+Y2)*(-torch.exp(-x-y)))
        elif cnvx_lss == 3:
            loss = -torch.mean((Y1+Y2)*(-torch.log(1+torch.exp(-x)+torch.exp(-y))))            
        else:
            loss = -torch.mean((Y1+Y2)*torch.max(torch.stack((x-1,y-1,torch.zeros_like(y)),dim=0),dim=0)[0])
            
    return loss

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def wavelet_feat(h):
    # Get wavelet approximations for a defined grid x:
    wavelet = pywt.Wavelet('db2')
    phi, psi, x = wavelet.wavefun(level=5)
    # all values will be between 0 and max (x)
    h = h - min(h)
    if max(h)  !=0 :
        h = h/max(h)*max(x)
    # Get wavelet fn approximations for H
    indxs = [find_nearest_idx(array=x, value=h_i) for h_i in h]
    phi_h = [phi[idx] for idx in indxs]
    psi_h = [psi[idx] for idx in indxs]
    #plt.plot(h, psi_h, 'o', color='black')
    return phi_h,psi_h

def WV_feats(H,J=5):
    H_WV = []
    for h in H.T:
        phi_h,psi_h = wavelet_feat(h) # phi and psi transformations
        H_WV.append(phi_h)
        H_WV.append(psi_h)
        for j in range(1,J+1):
            _,psi_h = wavelet_feat((2**j)*h-j)
            H_WV.append(psi_h)
    H_WV = np.array(H_WV).T
    # Add all column pairwise product
    col_prod = np.array([list(H_WV[:,i]*H_WV[:,j]) for  i,j in list(combinations(range(H_WV.shape[1]), 2))]).T
    H_WV = np.hstack((H_WV,col_prod))

    return np.hstack((H,H_WV))

def ns_feats(H):
    # Generating cubic spline with 3 knots at quintiles .25, .5 and .75
    ns_ls = []
    for j in range(H.shape[1]):
        q1,q2,q3 = np.quantile(H[:,j],.25),np.quantile(H[:,j],.5),np.quantile(H[:,j],.75)
        ns = dmatrix("bs(train, knots=(k1,k2,k3), degree=3, include_intercept=False)", {"train": H[:,j],"k1":q1,"k2":q2,"k3":q3},return_type='dataframe').T.values.tolist()
        ns_ls = ns_ls + ns
    return(np.array(ns_ls).T)

def batches(N, batch_size,seed):
    seq = [i for i in range(N)]
    random.seed(seed)
    if seed is not None:
        random.shuffle(seq)
    return (torch.tensor(seq[pos:pos + batch_size]) for pos in range(0, N, batch_size))


def pre_process(df,df_test,setting,f_model,O1_vars=None,O2_vars=None):
    if setting  == 'sepsis':
        O1,O2 = O1_vars,O2_vars
    elif setting == 1:
        O1,O2 = ['O1.1', 'O1.2', 'O1.3'],[]
    elif setting == 4:
        O1, O2 = ['O'+str(i) for i in range(1,7)], ['Z.21', 'Z.22']
    elif setting in [5,6]:
        O1, O2 = ['X1'], ['X1','X2']
    elif setting in list(range(10,14)):
        O1, O2 = ['X1','O11','O12'], ['X1','X2','O11','O12','O23']
    else:
        O1, O2 = ['O1.1', 'O1.2', 'O1.3'], ['O2.1', 'O2.2']

   
    Y1, Y1_test = df[['Y1']].to_numpy(), df_test[['Y1']].to_numpy()
    Y2, Y2_test = df[['Y2']].to_numpy(), df_test[['Y2']].to_numpy()
    A1, A1_test = df[['A1']].to_numpy(), df_test[['A1']].to_numpy()
    A2, A2_test = df[['A2']].to_numpy(), df_test[['A2']].to_numpy()
    H1, H1_test = df[O1].to_numpy(), df_test[O1].to_numpy()
    H2, H2_test = df[O1+['Y1']+O2].to_numpy(), df_test[O1+['Y1']+O2].to_numpy()    
    
    if 'SOWL' not in f_model:
        if f_model == 'wavelets':
            H1, H1_test = WV_feats(H1),WV_feats(H1_test)
            H2, H2_test = WV_feats(H2),WV_feats(H2_test)
        elif f_model == 'splines':
            H1, H1_test = ns_feats(H1),ns_feats(H1_test)
            H2, H2_test = ns_feats(H2),ns_feats(H2_test)

        # turn data into pytorch tensors
        Y1, Y1_test = Variable(torch.from_numpy(Y1)).float(), Variable(torch.from_numpy(Y1_test)).float()
        Y2, Y2_test = Variable(torch.from_numpy(Y2)).float(), Variable(torch.from_numpy(Y2_test)).float()
        A1, A1_test = Variable(torch.from_numpy(A1)).float(), Variable(torch.from_numpy(A1_test)).float()
        A2, A2_test = Variable(torch.from_numpy(A2)).float(), Variable(torch.from_numpy(A2_test)).float()
        H1, H1_test = Variable(torch.from_numpy(H1)).float(), Variable(torch.from_numpy(H1_test)).float()
        H2, H2_test = Variable(torch.from_numpy(H2)).float(), Variable(torch.from_numpy(H2_test)).float()
    
    return((Y1,Y2,A1,A2,H1,H2),(Y1_test, Y2_test, A1_test, A2_test, H1_test, H2_test))

def train_nnQs(f_model,tuple_train,learningRate = .1,epochs = 10,l1_reg=False,lamb=None,alpha=None):
    Y1,Y2,A1,A2,H1,H2 = tuple_train
    if f_model == 'NN':
        Q1 = net(H1.shape[1]+1, 1)
        Q2 = net(H2.shape[1]+1, 1)
    else: # linear or linear on wavelet basis
        Q1 = lm(H1.shape[1]+1, 1)
        Q2 = lm(H2.shape[1]+1, 1)            
    #
    ##### For GPU #######
    if torch.cuda.is_available():
        Q1.cuda()
        Q1.cuda()
    #
    Q1_optimizer = torch.optim.Adam(Q1.parameters(),lr=learningRate)
    Q2_optimizer = torch.optim.Adam(Q2.parameters(),lr=learningRate)
    #
    start_time = time.clock()
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for batch_indx in batches(N=H2.shape[0],batch_size=128,seed=epoch):
            H2_batch,Y2_batch,A2_batch = torch.index_select(H2, 0, batch_indx),torch.index_select(Y2, 0, batch_indx),torch.index_select(A2, 0, batch_indx)
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            Q2_optimizer.zero_grad()
            # get output from the model, given the inputs
            Q2_hat = Q2(torch.cat((H2_batch,A2_batch),dim=1))
            # get loss for the predicted output
            loss = loss_fn(Q2_hat, Y2_batch)
            # get gradients w.r.t to parameters
            loss.backward()
            # update parameters
            Q2_optimizer.step()
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for batch_indx in batches(N=H2.shape[0],batch_size=128,seed=epoch):
            H1_batch,H2_batch,Y1_batch,A1_batch = torch.index_select(H1, 0, batch_indx),torch.index_select(H2, 0, batch_indx),torch.index_select(Y1, 0, batch_indx),torch.index_select(A1, 0, batch_indx)
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            Q1_optimizer.zero_grad()            
            # get output from the model, given the inputs
            Q1_hat = Q1(torch.cat((H1_batch,A1_batch),dim=1))
            # Optimal treatment d2
            with torch.no_grad():
                Q2_negA2 = Q2(torch.cat((H2_batch,-torch.ones_like(A1_batch)),dim=1))
                Q2_posA2 = Q2(torch.cat((H2_batch,torch.ones_like(A1_batch)),dim=1))
                d2_hat = torch.argmax(torch.cat((Q2_negA2,Q2_posA2),dim=1), dim=1)
                d2_hat[d2_hat==0] = -1
            # get pseudo-outcome:            
            Q2_hat = Q2(torch.cat((H2_batch,d2_hat.unsqueeze(1).float()),dim=1))
            psuedo_Y1 = Y1_batch + Q2_hat
            # get loss for the predicted output
            loss = loss_fn(Q1_hat,psuedo_Y1)
            # get gradients w.r.t to parameters
            loss.backward()
            # update parameters
            Q1_optimizer.step()
    end_time = time.clock()
    return (Q1,Q2,end_time-start_time)

def Q_learning(tuple_train,tuple_test,f_model,learningRate = .1,epochs = 10,l1_reg=False,lamb=None,alpha=None):
    _, _, _, A2_test, H1_test, H2_test = tuple_test
    Q1,Q2,compute_time = train_nnQs(f_model,tuple_train,learningRate,epochs,l1_reg,lamb,alpha)    
    # Optimal treatments
    with torch.no_grad():
        Q2_negA2 = Q2(torch.cat((H2_test,-torch.ones_like(A2_test)),dim=1))
        Q2_posA2 = Q2(torch.cat((H2_test,torch.ones_like(A2_test)),dim=1))
        d2_hat = torch.argmax(torch.cat((Q2_negA2,Q2_posA2),dim=1), dim=1)
        d2_hat[d2_hat==0] = -1

        Q1_negA1 = Q1(torch.cat((H1_test,-torch.ones_like(A2_test)),dim=1))
        Q1_posA1 = Q1(torch.cat((H1_test,torch.ones_like(A2_test)),dim=1))
        d1_hat = torch.argmax(torch.cat((Q1_negA1,Q1_posA1),dim=1), dim=1)
        d1_hat[d1_hat==0] = -1
        
        d1_hat,d2_hat = d1_hat.detach().numpy(),d2_hat.detach().numpy()
    return(np.stack((d1_hat,d2_hat),axis=1),compute_time)

# Function to run K-fold cross-valildation for choosing optimal phi function for dataset
def CV_choose_phi(df,CV_K,setting,f_model,O1_vars = None,O2_vars = None,learningRate = .001,epochs = 10,l1_reg=False,lamb=None,alpha=None):
    print('Using CV for phi')
    phis = list(range(1,6))
    cv_res = []
    for phi_No in phis:
        kfolds = KFold(n_splits=CV_K, shuffle=True,random_state=116687)
        res_phi = []
        for train_indx,test_indx  in kfolds.split(range(df.shape[0])):
            cv_tuple_train,cv_tuple_test = pre_process(df.iloc[train_indx,:],df.iloc[test_indx,:],setting,f_model,O1_vars,O2_vars)
            cv_d_hat,_ = surr_opt(cv_tuple_train,cv_tuple_test,phi_No,f_model,learningRate,epochs,l1_reg,lamb,alpha)
            cv_d1_hat,cv_d2_hat = cv_d_hat[:,0],cv_d_hat[:,1]
            res_phi.append(eval_DTR(df.iloc[train_indx,:],df.iloc[test_indx,:],setting,cv_d1_hat,cv_d2_hat,O1_vars,O2_vars))
        cv_res.append(np.mean(res_phi))
    best_phi = np.nanargmax(cv_res)+1
    return best_phi

def surr_opt(tuple_train,tuple_test,phi_No,f_model,learningRate = .001,epochs = 10,l1_reg=False,lamb=None,alpha=None,cnvx_lss=None):
    Y1,Y2,A1,A2,H1,H2 = tuple_train
    _, _, _, _, H1_test, H2_test = tuple_test
    if f_model == 'NN':
        f1 = net(H1.shape[1], 1)
        f2 = net(H2.shape[1], 1)
    else: # linear or linear on wavelet basis
        f1 = lm(H1.shape[1], 1)
        f2 = lm(H2.shape[1], 1)            
    #
    ##### For GPU #######
    if torch.cuda.is_available():
        f1.cuda()
        f1.cuda()
    #
    #
    #optimizer = torch.optim.SGD(itertools.chain(f1.parameters(), f2.parameters()), lr=learningRate, momentum=0.9)    
    optimizer = torch.optim.RMSprop(itertools.chain(f1.parameters(), f2.parameters()), lr=learningRate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    #            
    start_time = time.clock()
    for epoch in range(epochs):
        for batch_indx in batches(N=H1.shape[0],batch_size=128,seed=epoch):
            H1_batch,H2_batch = torch.index_select(H1, 0, batch_indx),torch.index_select(H2, 0, batch_indx)
            Y1_batch,Y2_batch = torch.index_select(Y1, 0, batch_indx),torch.index_select(Y2, 0, batch_indx)
            A1_batch,A2_batch = torch.index_select(A1, 0, batch_indx),torch.index_select(A2, 0, batch_indx)
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()
            #
            # get output from the model, given the inputs
            f1_hat = f1(H1_batch)
            f2_hat = f2(H2_batch)
            #
            # get loss for the predicted output
            loss = my_loss(f1,f2,f1_hat,f2_hat,A1_batch,A2_batch,Y1_batch,Y2_batch,phi_No,l1_reg,lamb,alpha,cnvx_lss)
            # get gradients w.r.t to parameters
            loss.backward()
            #
            # update parameters
            optimizer.step()
            #
            #print('epoch {}, loss {}'.format(epoch, loss.item()))
    end_time = time.clock()
    #
    d1_hat,d2_hat= np.array([]), np.array([])
    with torch.no_grad():
    #for batch_indx in batches(N=H1_test.shape[0],batch_size=128*4,seed=None):
        d1_hat = np.append(d1_hat,np.sign(f1(H1_test).detach().numpy()))
        d2_hat = np.append(d2_hat,np.sign(f2(H2_test).detach().numpy()))
    #
    compute_time = end_time - start_time
    return(np.stack((d1_hat,d2_hat),axis=1),compute_time)
    

def simulations(size,setting,sims_No,f_model,learningRate = .1,epochs = 10,l1_reg=False,lamb=None,alpha=None,phi_No=None,CV_K=2,cnvx_lss=None):
    results = []
    test_size = int(1e5)
    #for sim in tqdm(range(sims_No)):
    for sim in range(sims_No):
        try: # BOWL method doesn't converge for some datasets
            #from utils import *
            #size,setting,sims_No,sim,f_model,test_size = 250,'disc',5,3,'BOWLradial',int(1e5)
            # Generate data:
            torch.manual_seed(sim)
            df = gen_data(size+test_size,setting,sim)
            df, df_test = df.loc[df.index[:size],:], df.loc[df.index[size:],:]
            tuple_train,tuple_test = pre_process(df,df_test,setting,f_model)
            #  Estimate treatment regime:
            if 'SOWL' in f_model:
                phi_No = None
                d_hat,compute_time = SOWL_optim(tuple_train,tuple_test,f_model,gamma = .01)# for toy .0005
            elif 'BOWL' in f_model:
                phi_No = None
                d_hat,compute_time = BOWL_optim(df,df_test,setting,O1_vars = None,O2_vars = None, surr_fn =  'hinge',kernel = f_model[4:])                                          
            elif f_model == 'DQlearning':
                phi_No = None
                d_hat,compute_time = Q_learning(tuple_train,tuple_test,f_model='NN',learningRate=learningRate,epochs=epochs)
            elif f_model == 'linQlearning':
                phi_No = None
                d_hat,compute_time = Q_learning(tuple_train,tuple_test,f_model='linear',learningRate=learningRate,epochs=epochs)
            else:
                phi_No,orig_phi = CV_choose_phi(df,CV_K,setting,f_model) if phi_No is None else phi_No,phi_No
                d_hat,compute_time = surr_opt(tuple_train,tuple_test,phi_No,f_model,learningRate,epochs,l1_reg,lamb,alpha,cnvx_lss)
                phi_No = orig_phi

            d1_hat,d2_hat = d_hat[:,0],d_hat[:,1]
            # Evaluate estimated regime:
            d1_err = np.mean(d1_hat.squeeze() != df_test['d1.star'].to_numpy())
            d2_err = np.mean(d2_hat.squeeze() != df_test['d2.star'].to_numpy())
            Rx_dict = {(1,1):['p1p1',0],(-1,1):['n1p1',1],(1,-1):['p1n1',2],(-1,-1):['n1n1',3]}
            Vs_np = df_test.loc[:,['p1p1','n1p1','p1n1','n1n1']].to_numpy()
            V_of_DTR_hat_ls = [Vs_np[i,Rx_dict[(int(d1_hat[i]),int(d2_hat[i]))][1]] for i in range(len(Vs_np))]
            V_of_DTR_hat,std_V_of_DTR_hat = np.mean(V_of_DTR_hat_ls),np.std(V_of_DTR_hat_ls)
            #
            V_of_DTR_star_ls = df_test.loc[:,['p1p1','n1p1','p1n1','n1n1']].max(1)
            V_of_DTR_star, std_V_of_DTR_star = np.mean(V_of_DTR_star_ls),np.std(V_of_DTR_star_ls)
            #
            V_hat_of_DTR_hat = eval_DTR(df,df_test,setting,d1_hat,d2_hat)
            results.append([d1_err,d2_err,compute_time,V_of_DTR_hat,V_hat_of_DTR_hat,V_of_DTR_star,std_V_of_DTR_hat,std_V_of_DTR_star])
            #
            pd_results = pd.DataFrame(results, columns = ['d1_err','d2_err','time','V_of_DTR_hat','V_hat_of_DTR_hat','V_of_DTR_star','std_V_of_DTR_hat','std_V_of_DTR_star']) 
            print('../Results/'+f_model+'_size_'+str(size)+'_setting_'+str(setting)+'_phi_No_'+str(phi_No)+'_sims_No_'+str(sims_No)+'_cnvx_lss_'+str(cnvx_lss)+'.csv')
            if os.getcwd()[:3]=='/n/':
                pd_results.to_csv('../Results/'+f_model+'_size_'+str(size)+'_setting_'+str(setting)+'_phi_No_'+str(phi_No)+'_sims_No_'+str(sims_No)+'_cnvx_lss_'+str(cnvx_lss)+'.csv')
            sys.stderr.write('\n'+f_model+'_size_'+str(size)+'_setting_'+str(setting)+'_phi_No_'+str(phi_No)+'_sims_No_'+str(sims_No)+'\n')
            sys.stderr.write('sim='+str(sim)+'\n'+str(pd_results.mean())+str(pd_results.shape))
        except:
            pass             
    return results

def eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars=None,O2_vars=None):
    #########################
    ### Propensity Scores ###
    #########################
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # Numpy format data
    np_tuple_train,np_tuple_test = pre_process(df,df_test,setting,f_model='SOWL',O1_vars=O1_vars,O2_vars=O2_vars)
    _,_,A1,A2,H1,H2 = np_tuple_train
    Y1_test,Y2_test,A1_test,A2_test,H1_test,H2_test = np_tuple_test

    #logreg = LogisticRegression(penalty='l2',C=1)    
    logreg = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2',C=.01))
    pi1 = logreg.fit(H1,np.ravel(A1))
#    A1zero= A1
#    A1zero[A1zero==-1]=0
#    (np.mean((A1zero-pi1.predict_proba(H1)[:,1])**2))**.5
    omega1,n_subpop1 = SelfNorm_omega_t(d_t=d1_hat,pi_t=pi1,A_t=A1_test.squeeze(),H_t=H1_test)
    logreg = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2',C=.01))
    pi2 = logreg.fit(H2,np.ravel(A2))
#    A2zero= A2
#    A2zero[A2zero==-1]=0
#    (np.mean((A2zero-pi2.predict_proba(H2)[:,1])**2))**.5
    omega2,n_subpop2 = SelfNorm_omega_t(d_t=d2_hat,pi_t=pi2,A_t=A2_test.squeeze(),H_t=H2_test,omega_tm1=omega1)

    ###################
    # Q-functions
    # Numpy format data 
    torch_tuple_train,torch_tuple_test = pre_process(df,df_test,setting,f_model='NN',O1_vars=O1_vars,O2_vars=O2_vars)
    _, _, _, _, H1_test, H2_test = torch_tuple_test
    Q1,Q2,_ = train_nnQs('NN',torch_tuple_train,learningRate = .01,epochs = 20,l1_reg=False,lamb=None,alpha=None)
    with torch.no_grad():
        Q1_opt = Q1(torch.cat((H1_test,torch.from_numpy(d1_hat).unsqueeze(1).float()),dim=1)).detach().numpy().squeeze()
        Q2_opt = Q2(torch.cat((H2_test,torch.from_numpy(d2_hat).unsqueeze(1).float()),dim=1)).detach().numpy().squeeze()
    
    subPop = [p1 and p2 for p1,p2 in zip(n_subpop1,n_subpop2)]
    
    Vfun = Q1_opt + omega1*(Y1_test.squeeze()-Q1_opt+Q2_opt) + omega2*(Y2_test.squeeze()-Q2_opt)
    
    Vfun = np.array([V if include else 0 for V,include in zip(Vfun,subPop)])
    
    #np.mean(omega1*(Y1_test.squeeze()-Q1_opt+Q2_opt))
    #np.mean(omega2*(Y2_test.squeeze()-Q2_opt))
    #np.mean(Q1_opt)
    return sum(Vfun)/sum(subPop)#np.mean(Vfun)

def SelfNorm_omega_t(d_t,pi_t,A_t,H_t,omega_tm1=None):
    A_t[A_t==0]=-1
    pi_negA,pi_posA = pi_t.predict_proba(H_t)[:,0],pi_t.predict_proba(H_t)[:,1]
    
    SN_omega = (d_t==A_t)*((A_t==1)/pi_posA+(A_t==-1)/pi_negA)
    # sub-population
    subPOP = [True if (piN < 0.99 and piN >0.01) and (piP < 0.99 and piP >0.01) else False for piN,piP in zip(pi_negA,pi_posA)]
    SN_omega = np.array([w if belongs else 0 for w,belongs in zip(SN_omega,subPOP)])

    #pi_negA = np.array([0.99 if pi  > 0.99 else .01 if pi <  0.01 else pi for pi in pi_negA])
    #pi_posA = np.array([0.99 if pi  > 0.99 else .01 if pi <  0.01 else pi for pi in pi_posA])
    
    #SN_Wneg1 = (1/pi_negA)/sum(1/pi_negA)#P(A=-1|H)
    #SN_Wpos1 = (1/pi_posA)/sum(1/pi_posA)#P(A=1|H)
    #SN_omega = (d_t==A_t)*((A_t==1)*SN_Wpos1+(A_t==-1)*SN_Wneg1)

    if omega_tm1 is not None:
        SN_omega *= omega_tm1
    return SN_omega, subPOP

## Run all methods on a data set

def run_DTRs(df,train_IDs,test_IDs,O1_vars,O2_vars,seed):
    random.seed(seed)

    # Split data into train & test samples
    df, df_test = df.loc[train_IDs,:], df.loc[test_IDs,:]

    setting = 'sepsis'
    res = {}

    # Surrogate function methods will choose phi based on 10-fold CV
    CV_K = 5
    if True:
        f_model = 'linQlearning'
        print(f_model)
        tuple_train,tuple_test = pre_process(df,df_test,setting,f_model,O1_vars,O2_vars)
        d_hat_Ql,compute_time_Ql = Q_learning(tuple_train,tuple_test,f_model='NN',learningRate = .01,epochs = 20)
        d1_hat,d2_hat = d_hat_Ql[:,0],d_hat_Ql[:,1]
        result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
        res[f_model] = result
        print(np.mean(d_hat_Ql,0))
        print(result)

        f_model = 'DQlearning'
        print(f_model)
        tuple_train,tuple_test = pre_process(df,df_test,setting,f_model,O1_vars,O2_vars)
        d_hat_Ql,compute_time_Ql = Q_learning(tuple_train,tuple_test,f_model='NN',learningRate = .01,epochs = 20)
        d1_hat,d2_hat = d_hat_Ql[:,0],d_hat_Ql[:,1]
        result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
        res[f_model] = result
        print(np.mean(d_hat_Ql,0))
        print(result)
        
        f_model ='linear'
        print(f_model)
        phi_No = CV_choose_phi(df,CV_K,setting,f_model,O1_vars,O2_vars)
        tuple_train,tuple_test = pre_process(df,df_test,setting,f_model,O1_vars,O2_vars)
        d_hat_LR,compute_time_LR = surr_opt(tuple_train,tuple_test,phi_No,f_model)
        d1_hat,d2_hat = d_hat_LR[:,0],d_hat_LR[:,1]
        result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
        res[f_model] = result
        print(np.mean(d_hat_LR,0))
        print(result)

        f_model = 'NN'
        print(f_model)
        phi_No = CV_choose_phi(df,CV_K,setting,f_model,O1_vars,O2_vars)
        tuple_train,tuple_test = pre_process(df,df_test,setting,f_model,O1_vars,O2_vars)
        d_hat_NN,compute_time_NN = surr_opt(tuple_train,tuple_test,phi_No,f_model)
        d1_hat,d2_hat = d_hat_NN[:,0],d_hat_NN[:,1]
        result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
        res[f_model] = result
        print(np.mean(d_hat_NN,0))
        print(result)

        f_model = 'splines'
        print(f_model)
        phi_No = CV_choose_phi(df,CV_K,setting,f_model,O1_vars,O2_vars)
        tuple_train,tuple_test = pre_process(df,df_test,setting,f_model,O1_vars,O2_vars)
        d_hat_CS,compute_time_CS = surr_opt(tuple_train,tuple_test,phi_No,f_model)
        d1_hat,d2_hat = d_hat_CS[:,0],d_hat_CS[:,1]
        result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
        res[f_model] = result
        print(np.mean(d_hat_CS,0))
        print(result)

        f_model ='wavelets'
        print(f_model)
        phi_No = CV_choose_phi(df,CV_K,setting,f_model,O1_vars,O2_vars)
        tuple_train,tuple_test = pre_process(df,df_test,setting,f_model,O1_vars,O2_vars)
        d_hat_WV,compute_time_WV = surr_opt(tuple_train,tuple_test,phi_No,f_model)
        d1_hat,d2_hat = d_hat_WV[:,0],d_hat_WV[:,1]
        result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
        res[f_model] = result
        print(np.mean(d_hat_WV,0))
        print(result)

        f_model ='SOWLRBF'
        print(f_model)
        tuple_train,tuple_test = pre_process(df,df_test,setting,f_model,O1_vars,O2_vars)
        d_hat_SOWL,compute_time_SOWL = SOWL_optim(tuple_train,tuple_test,f_model,gamma = 1)
        d1_hat,d2_hat = d_hat_SOWL[:,0],d_hat_SOWL[:,1]
        result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
        res[f_model] = result
        print(np.mean(d_hat_SOWL,0))
        print(result)

        f_model ='SOWLlinear'
        print(f_model)
        tuple_train,tuple_test = pre_process(df,df_test,setting,f_model,O1_vars,O2_vars)
        d_hat_SOWL,compute_time_SOWL = SOWL_optim(tuple_train,tuple_test,f_model,gamma = 1)
        d1_hat,d2_hat = d_hat_SOWL[:,0],d_hat_SOWL[:,1]
        result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
        res[f_model] = result
        print(np.mean(d_hat_SOWL,0))
        print(result)
    if True:
        try:
            f_model ='BOWLlinear'                                   
            print(f_model)
            d_hat_BOWLlinear,compute_time_BOWLlinear = BOWL_optim(df,df_test,setting,O1_vars,O2_vars, surr_fn =  'hinge',kernel = f_model[4:])
            d1_hat,d2_hat = d_hat_BOWLlinear[:,0],d_hat_BOWLlinear[:,1]
            result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
            res[f_model] = result
            print(np.mean(d_hat_BOWLlinear,0))
            print(result)
        except:
            print('pass')
            #pass
        
        try:    
            f_model ='BOWLradial'                                   
            print(f_model)
            d_hat_BOWLradial,compute_time_BOWLradial = BOWL_optim(df,df_test,setting,O1_vars,O2_vars, surr_fn =  'hinge',kernel = f_model[4:])
            d1_hat,d2_hat = d_hat_BOWLradial[:,0],d_hat_BOWLradial[:,1]
            result = eval_DTR(df,df_test,setting,d1_hat,d2_hat,O1_vars,O2_vars)
            res[f_model] = result
            print(np.mean(d_hat_BOWLradial,0))
            print(result)
        except:
            print('pass')
            #pass

    return(res)


##############################
########## SOWL ##############
##############################

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

def SOWL_optim(tuple_train,tuple_test,f_model,gamma = 1):
    Y1,Y2,A1,A2,H1,H2 = tuple_train
    _, _, _, _, H1_test, H2_test = tuple_test
    n = H2.shape[0]
    ##########
    ########## Objective function: Min (1/2)alpha^T * P * alpha + q^T alpha
    ##########
    ##### P matrix
    # Compute Kernel matrices
    if f_model == 'SOWLRBF':
        K1 = rbf_kernel(H1) 
        K2 = rbf_kernel(H2)
    elif f_model == 'SOWLlinear':
        K1 = linear_kernel(H1) 
        K2 = linear_kernel(H2)
    # Compute treatment matrices where A1A1t_(i,j) = AiAj
    A1A1t = A1.dot(A1.T)
    A2A2t = A2.dot(A2.T)
    # Elementwise (Kronecker) product:
    A1A1K1 = np.multiply(A1A1t,K1)
    A2A2K2 = np.multiply(A2A2t,K2)
    # Large weight matrix: for (alpha^T)*P*alpha
    P = np.block([
        [A1A1K1,                 np.zeros(A1A1K1.shape)],
        [np.zeros(A1A1K1.shape), A2A2K2               ]
        ])
    #Converting into cvxopt format
    P = cvxopt_matrix(P)
    ##### q vector
    # This vector is negative as we're minimizing in the cvxopt objective function
    q = cvxopt_matrix(-np.ones((2*n, 1)))
    ##########
    ########## Inequality constraint G * alpha <= h
    ##########
    ##### G matrix
    # G matrix such that G*alpha = (alpha1+alpha2,-alpha1,-alhpa2)
    G = np.block([[np.eye(n),np.eye(n)],
                [-np.eye(n),np.zeros((n,n))],
                [np.zeros((n,n)),-np.eye(n)]])
    #Converting into cvxopt format
    G = cvxopt_matrix(G)              
    # Weight vector (still need to adjust by propensity score)
    pi1,pi2 = .5, .5
    W = (Y1+Y2)/(pi1*pi2)
    ##### h vector
    # h vector such that G*alpha <= h where h = (gamma*W,0,0)
    h = cvxopt_matrix(np.vstack((gamma*W,np.zeros((2*n,1)))))
    ##########
    ########## Equality constraint A * alpha = b
    ##########
    ##### A matrix
    # A matrix such that A * alpha = (A1^T * alpha1, A2^T * alpha2)
    A = np.block([[A1.T,np.zeros((A1.T.shape))],
                [np.zeros((A2.T.shape)),A2.T]])
    #Converting into cvxopt format
    A = cvxopt_matrix(A)              
    ##### b matrix
    b = cvxopt_matrix(np.zeros(2))
    #Setting solver parameters (change default to decrease tolerance) 
    cvxopt_solvers.options['show_progress'] = False
    #cvxopt_solvers.options['abstol'] = 1e-10
    #cvxopt_solvers.options['reltol'] = 1e-10
    #cvxopt_solvers.options['feastol'] = 1e-10
    #Run solver
    start_time = time.clock()
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    end_time = time.clock()
    alphas = np.array(sol['x'])
    #==================Test===============================#
    # Select support vectors
    S = (alphas > 1e-4).flatten()
    alph_indx1 = S[:n]
    alph_indx2 = S[n:]
    # alpha_k*A_k, k=1,2 vectors
    A1alph1 = (A1*alphas[:n])[alph_indx1]
    A2alph2 = (A2*alphas[n:])[alph_indx2]
    # RBF kernel matrtices
    if f_model == 'SOWLRBF':
        K1_test = rbf_kernel(H1,H1_test)[alph_indx1,:]
        K2_test = rbf_kernel(H2,H2_test)[alph_indx2,:]
    elif f_model == 'SOWLlinear':
        K1_test = linear_kernel(H1,H1_test)[alph_indx1,:]
        K2_test = linear_kernel(H2,H2_test)[alph_indx2,:]
    #
    # Remove zeros
    #d1_hat,d2_hat = 2*(d1_hat >= 0) - 1,2*(d2_hat >= 0) - 1
    #d1_hat = np.sign(A1alph1.T.dot(K1_test)).squeeze() 
    #d2_hat = np.sign(A2alph2.T.dot(K2_test)).squeeze()
    
    d1_hat = 2*(np.sign(A1alph1.T.dot(K1_test)).squeeze()  >= 0) - 1
    d2_hat = 2*(np.sign(A2alph2.T.dot(K2_test)).squeeze() >= 0) - 1
    compute_time = end_time-start_time
    return(np.stack((d1_hat,d2_hat),axis=1),compute_time)




