################################################################
########################## Functions ###########################
################################################################
library(splines)
library(dplyr)
library(MASS)
library(DynTxRegime)
library(wavelets)

# 
# # obtain the two series listed in Percival and Walden (2000), page 42
# X1 <- c(.2,-.4,-.6,-.5,-.8,-.4,-.9,0,-.2,.1,-.1,.1,.7,.9,0,.3)
# X2 <- c(.2,-.4,-.6,-.5,-.8,-.4,-.9,0,-.2,.1,-.1,.1,-.7,.9,0,.3)
# # combine them and compute DWT
# newX <- cbind(X1,X2)
# wt <- dwt(newX, n.levels=4,  filter="d12", boundary="reflection", fast=FALSE)
# wt@V[1]
# wt@V[2]
# wt@V[3]
# wt@V[4]
# 
# # father es scaling
# 
# ####
# wt.filter.shift(filter, J, wavelet=TRUE, coe=FALSE, modwt=FALSE)

quiet <- function(x) { 
  sink(tempfile()) 
  on.exit(sink()) 
  invisible(force(x)) 
} 
expit <- function(x) exp(x)/(1+exp(x))

gen_data <- function(size,setting,seed){
  set.seed(seed)
  O1_dim <- 3
  #50-dimensional baseline covariatesX1,1, . . . , X1,50 are generated according to N(0, 1).
  O1 <- matrix(rnorm(size*O1_dim,0,1),nrow=size,O1_dim)
  
  #O1 <- cbind(rbinom(size,1,.2),rbinom(size,10,.4),rbinom(size,10,.8))
  
  # Treatments A1, A2 are randomly generated from {−1, 1} with equal probability 0.5.
  trt <- rbinom(2*size,1,.5); trt[trt==0] <- -1
  A1 <- trt[1:size]; A2 <- trt[(size+1):(2*size)]
  
  # The models for generating outcomes R1 and R2 vary under the different settings stated below:
  # Setting 1) 
  if(setting==1){
    c1 <- 3;c2 <- 15
    # Stage 1 outcome Y1 is generated according to: N(0.5*X_13*A1, 1), 
    Y1_mean <- function(O1,A1) .5*O1[,3]*A1 + c1
    Y1 <- Y1_mean(O1,A1) + rnorm(size,0,1)
    # and Stage 2 outcome Y2 is generated according to N((((X_11)^2 + (X_12)^2 − 0.2)(0.5 − (X_11)^2 − (X_12)^1) + Y1)*A2, 1).
    Y2_mean <- function(O1,Y1,A2) ((O1[,1]^2 + O1[,2]^2 - 0.2)*(0.5 - O1[,1]^2 - O1[,2]^2) + Y1)*A2 + c2
    Y2 <- Y2_mean(O1,Y1,A2) + rnorm(size,0,1)
    O2.1 <- O2.2 <- rep(NA,size)
    taus <- cbind(p1p1=Y1_mean(O1,A1=1)+Y2_mean(O1,Y1,A2=1),
                  n1p1=Y1_mean(O1,A1=-1)+Y2_mean(O1,Y1,A2=1),
                  p1n1=Y1_mean(O1,A1=1)+Y2_mean(O1,Y1,A2=-1),
                  n1n1=Y1_mean(O1,A1=-1)+Y2_mean(O1,Y1,A2=-1))
  }else if(setting==2){### This is setting 3 in overleaf
    cnst <- 10
    # Setting 2) 
    # Stage 1 outcome Y1 is generated according to: N((1 +1.5X_13)*A1, 1);
    Y1_mean <- function(O1,A1) (1 +1.5*O1[,3])*A1 + cnst
    #Y1_mean <- function(O1,A1) O1[,3]*A1 + cnst
    Y1 <- Y1_mean(O1,A1) + rnorm(size,0,1)
    # two intermediate variables, O2,1 ∼ I {N(1.25*X_11*A1, 1) > 0}, and O_22 ∼ I {N(−1.75*O_12*A1, 1) > 0} are generated;
    O2.1 <- as.numeric(1.25*O1[,1]*A1+rnorm(size,0,1)>0); O2.2 <- as.numeric(-1.75*O1[,2]*A1+rnorm(size,0,1)>0)
    # then the Stage 2 outcome Y2 is generated according to N((0.5 + Y1 + 0.5*A1 +0.5*X_21 − 0.5*X2,2)*A2, 1).
    Y2_mean <- function(O1,O2.1,O2.2,Y1,A2) (-.5 + 0.5*Y1 + 0.5*A1 +0.5*O2.1 - 0.5*O2.2)*A2 + cnst# previous one (Zhang et al)
    #Y2_mean <- function(O1,Y1,A2) -O1[,2]*A2 + cnst
    Y2 <- Y2_mean(O1,O2.1,O2.2,Y1,A2) + rnorm(size,0,1)
    taus <- cbind(p1p1=Y1_mean(O1,A1=1)+Y2_mean(O1,O2.1,O2.2,Y1,A2=1),
                  n1p1=Y1_mean(O1,A1=-1)+Y2_mean(O1,O2.1,O2.2,Y1,A2=1),
                  p1n1=Y1_mean(O1,A1=1)+Y2_mean(O1,O2.1,O2.2,Y1,A2=-1),
                  n1n1=Y1_mean(O1,A1=-1)+Y2_mean(O1,O2.1,O2.2,Y1,A2=-1))
    
  }else if (setting == 3){
    O1[,1] <- O1[,1]/10
    # Setting 2) 
    # Stage 1 outcome Y1 is generated according to: 
    Y1_mean <- function(O1,A1) (1 +1.5*O1[,3]>0)*A1 + 2#==(1+1.5*O1[,3]>0)*A1
    #Y1_mean <- function(O1,A1) O1[,3]*A1 + cnst
    Y1 <- Y1_mean(O1,A1) + rnorm(size,0,1)
    # then the Stage 2 outcome Y2 is generated according to:
    O2.1 <- rnorm(size,0,1); O2.2 <- rep(0,size)
    Y2_mean <- function(O1,O2.1,Y1,A2) 10*A2*sign(.1*O1[,2]^2-.05*O1[,3]^2)+O2.1 + Y1_mean(O1,A1) 
    Y2 <- Y2_mean(O1,O2.1,Y1,A2) + rnorm(size,0,1)
    taus <- cbind(p1p1=Y1_mean(O1,A1=1)+Y2_mean(O1,O2.1,Y1,A2=1),
                  n1p1=Y1_mean(O1,A1=-1)+Y2_mean(O1,O2.1,Y1,A2=1),
                  p1n1=Y1_mean(O1,A1=1)+Y2_mean(O1,O2.1,Y1,A2=-1),
                  n1n1=Y1_mean(O1,A1=-1)+Y2_mean(O1,O2.1,Y1,A2=-1))
    
  }else if (setting == 4){
      # parameters:
      beta26 <- 0 #Q func. missp.
      phi24 <- 0 #prop. score missp. 
      ########
      # A1:
      phi1 <- c(.3,-.5)
      names(phi1) <- c('10','11')
      # Y1:
      beta1 <- c(1,1,1,-2)#beta1 <- c(1,1,1,1)#
      names(beta1) <- c(10:13)
      # O2:
      delta1 <- c(0, .5,-.75, .25,1)
      names(delta1) <- c(10:14)
      
      #A2:
      phi2 <- c(0, .5, .1,-1,phi24,-.1,1)
      names(phi2) <- c(20:26)
      
      #Y2
      beta2 <- c(3, 0, .1,-.5,-.5,.1,beta26)
      psi2 <- c(1, .25, .5)#psi2 <- c(1, .25, 1.5)#
      names(beta2) <- c(20:26)
      names(psi2) <- c(20:22)
      ########
      #set.seed(seed)
      # True Q function working models:
      S1 <- ~ O1+O2+O3+O4+O5+O6+A1*((O2<O3^2)*(abs(O4)>O5)); S2 <- ~ Y1+O1+O2+O3+O4+O5+O6+A1+Z.21+Z.21+Z.22+A2*((O1>O2)*(O3>O6)+(abs(O4)<1)+Z.21)# high dimension
      # True propensity score functions:
      PS1 <- ~ O1+O2+O3; PS2 <- ~ Y1+O1+A1+Z.21+O2
      
      O.var = diag(.9,6,6)+.1
      O <- data.frame(mvrnorm(n = size, mu=rep(0,6), Sigma=O.var, tol = 1e-6, empirical = FALSE, EISPACK = FALSE))
      #O[,1:4] <- floor(O[,1:4])
      #O <- floor(O)
      colnames(O) <- gsub('X','O',colnames(O))
      H1check <- model.matrix.lm(PS1,O)
      phi1 <- c(-.1,1,-1,.1)
      prob_A1 <- expit(tcrossprod(t(phi1),H1check))
      #A1 <- rbinom(size,1,prob_A1)
      #A1[A1==0] <- -1
      beta1 <- c(.5,.2,-1,-1,.1,-.1,.1)
      gamma1 <- 10*c(1,-2,-2,-.1,.1,-1.5)
      
      H1.Q.p <- model.matrix.lm(S1,cbind(O,A1=1))
      mean_Y1.p <- t(tcrossprod(t(c(beta1,gamma1,.1)),H1.Q.p))+3
      H1.Q.n <- model.matrix.lm(S1,cbind(O,A1=-1))
      mean_Y1.n <- t(tcrossprod(t(c(beta1,gamma1,.1)),H1.Q.n))+3
      
      H1.Q <- model.matrix.lm(S1,cbind(O,A1=A1))
      mean_Y1 <- t(tcrossprod(t(c(beta1,gamma1,.1)),H1.Q))+3
      Y1 <- mean_Y1 + rnorm(size,0,1) #rbinom(size,1,expit(mean_Y1)) 
      
      
      Z.21 <- as.numeric(1.25*O[,1]*A1+rnorm(n=size,mean=0,sd=1) >0)
      Z.22 <- as.numeric(-1.75*O[,2]*A1+rnorm(n=size,mean=0,sd=1) >0)
      
      H2check <- model.matrix.lm(PS2,data.frame(O1=O[,1],O2=O[,2],Y1=Y1,A1=A1,Z.21=Z.21))
      mean_A2 <- tcrossprod(t(phi2[1:ncol(H2check)]),H2check)
      if (phi24 != 0){
        # model + a misspecified term which is positive when phi24 is different than zero
        phi24.vec <- t(c(phi24,phi24)/as.numeric(sqrt(crossprod(c(phi24,phi24)))))
        mean_A2 <- mean_A2 + tcrossprod(phi24.vec,H2check[,c('Y1','Z.21')])*Y1*sin(apply(H2check[,c('Y1','Z.21')],1,crossprod)/(Y1+1))
      }
      prob_A2 <- expit(mean_A2)
      #A2 <- rbinom(size,1,prob_A2)
      #A2[A2==0] <- -1
      beta2 <- c(1,beta1,.25,-1,-.5)
      gamma2 <- 10*c(1,.1,-.1,.1,-.1,.25,-1,-.5)
      H2.Q <- model.matrix.lm(S2,cbind(O,A1=A1,A2=A2,Y1,Z.21,Z.22))
      mean_Y2 <- t(tcrossprod(t(c(beta2,gamma2,.1,.1)),H2.Q))+3
      
      H2.Q.p <- model.matrix.lm(S2,cbind(O,A1=A1,A2=1,Y1,Z.21,Z.22))
      mean_Y2.p <- t(tcrossprod(t(c(beta2,gamma2,.1,.1)),H2.Q.p))+3
      H2.Q.n <- model.matrix.lm(S2,cbind(O,A1=A1,A2=-1,Y1,Z.21,Z.22))
      mean_Y2.n <- t(tcrossprod(t(c(beta2,gamma2,.1,.1)),H2.Q.n))+3
      
      if (beta26 != 0){
        # model + a misspecified term which is positive when beta26 is different than zero
        beta26.vec <- t(c(beta26,beta26)/as.numeric(sqrt(crossprod(c(beta26,beta26)))))
        mean_Y2 <- mean_Y2 + t(tcrossprod(beta26.vec,H2.Q[,c('Z.21','Z.22')]))*Y1*sin(apply(H2.Q[,c('Z.21','Z.22')],1,crossprod)/(Y1+1))
      }
      Y2 <- mean_Y2 + rnorm(size,0,1) #rbinom(size,1,expit(mean_Y2))
      taus <- data.frame(p1p1=mean_Y1.p+mean_Y2.p,
                         n1p1=mean_Y1.n+mean_Y2.p,
                         p1n1=mean_Y1.p+mean_Y2.n,
                         n1n1=mean_Y1.n+mean_Y2.n)
  }else if (setting == 5){
    X1 <- rnorm(size,0,1);X2 <- rnorm(size,0,1)
    Y1_mean <- function(X1,A1) A1*(2*(abs(X1)<1)-1)
    Y2_mean <- function(X1,X2,A2) A2*(2*(X2-X1^2>0)-1)
    
    Y1 <- Y1_mean(X1,A1) + rnorm(size,0,1)
    Y2 <- Y2_mean(X1,X2,A2) + rnorm(size,0,1)
    taus <- cbind(p1p1=Y1_mean(X1,A1=1)+Y2_mean(X1,X2,A2=1),
                  n1p1=Y1_mean(X1,A1=-1)+Y2_mean(X1,X2,A2=1),
                  p1n1=Y1_mean(X1,A1=1)+Y2_mean(X1,X2,A2=-1),
                  n1n1=Y1_mean(X1,A1=-1)+Y2_mean(X1,X2,A2=-1))
    
  }else if (setting == 6){
    
    X <- rbinom(2*size,1,.5);X[X==0] <- -1
    X1 <- X[1:size];X2 <- X[(size+1):(2*size)]
    Y1_mean <- 1
    Y2_mean <- function(A1,A2) (A1==1 & A2==1)*4 + (A1==1 & A2==-1)*3 + (A1==-1 & A2==1)*5 + (A1==-1 & A2==-1)*1
    
    Y1 <- Y1_mean + rnorm(size,0,1)
    Y2 <- Y2_mean(A1,A2) + rnorm(size,0,1)
    taus <- cbind(p1p1=Y1_mean+Y2_mean(A1=1,A2=1),
                  n1p1=Y1_mean+Y2_mean(A1=-1,A2=1),
                  p1n1=Y1_mean+Y2_mean(A1=1,A2=-1),
                  n1n1=Y1_mean+Y2_mean(A1=-1,A2=-1))
    
  }else if (setting %in% c(10:13)){
    # Covariates
    X1 <- rnorm(size,0,.1);O11 <- O1[,1]/10;O12 <- O1[,2]/10
    X2 <- rnorm(size,0,.1);O23 <- (O1[,1]+3)^2*A1/10
    
    #X1 <- abs(X1);O11 <- abs(O11);O12 <- abs(O12)
    #X2 <- abs(X2);O23 <- abs(O23)
    
    # Mean functions
    m01_hard <- function(X1,O11,O12) (2*(abs(X1)<1)+1) + (.1*O12^2+.05*O11^2)
    m02_hard <- function(X1,X2,O11,O12,O23) (2*(X2-X1^2>0)+1) + (.1*O12^2+.05*O11^2) +.1*O23
    m11_hard <- function(X1,O11,O12,A1) (A1+1)*(2*(abs(X1)<1)+1) + (A1+1)*(.3*O12+.2*O11)
    m12_hard <- function(X1,X2,O11,O12,O23,A2) (A2+1)*(2*(X2-X1^2>0)-1) + (A1+1)*(.3*O12+.2*O11) +.1*(A2+1)*O23
    
    m01_easy <- function(X1,O11,O12) 2*X1+1 + (.1*O12^2+.05*O11^2)
    m02_easy <- function(X1,X2,O11,O12,O23) 2*X2+.5*X1^2+1 + (.1*O12^2+.05*O11^2) +.1*O23
    m11_easy <- function(X1,O11,O12,A1) 2*(A1+1)*(X1+1) + (A1+1)*(.3*O12+.2*O11)
    m12_easy <- function(X1,X2,O11,O12,O23,A2) (A2+1)*.2*(X2+X1^2+1) + (A1+1)*(.3*O12+.2*O11) +.1*(A2+1)*O23
    
    if (setting ==10){
      Y1_mean <- function(X1,O11,O12,A1) m01_easy(X1,O11,O12) + m11_easy(X1,O11,O12,A1) + .5
      Y2_mean <- function(X1,X2,O11,O12,O23,A2) m02_easy(X1,X2,O11,O12,O23) + m12_easy(X1,X2,O11,O12,O23,A2) + .5
    }else if (setting ==11){
      Y1_mean <- function(X1,O11,O12,A1) m01_hard(X1,O11,O12) + m11_easy(X1,O11,O12,A1) + .5
      Y2_mean <- function(X1,X2,O11,O12,O23,A2) m02_hard(X1,X2,O11,O12,O23) + m12_easy(X1,X2,O11,O12,O23,A2) + .5
    }else if (setting ==12){
      Y1_mean <- function(X1,O11,O12,A1) m01_easy(X1,O11,O12) + m11_hard(X1,O11,O12,A1) + .5
      Y2_mean <- function(X1,X2,O11,O12,O23,A2) m02_easy(X1,X2,O11,O12,O23) + m12_hard(X1,X2,O11,O12,O23,A2) + .5
    }else{#setting=13
      Y1_mean <- function(X1,O11,O12,A1) m01_hard(X1,O11,O12) + m11_hard(X1,O11,O12,A1) + .5
      Y2_mean <- function(X1,X2,O11,O12,O23,A2) m02_hard(X1,X2,O11,O12,O23) + m12_hard(X1,X2,O11,O12,O23,A2) + .5
    } 
    
    #1) spline vs linear is worrisome
    #2) use a single phi for simulations
    #3) take functions that we use in settings 1-5 to make the combination simulations
    #4) Double check the SGD
    
    Y1 <- Y1_mean(X1,O11,O12,A1) + rnorm(size,0,1)
    Y2 <- Y2_mean(X1,X2,O11,O12,O23,A2) + rnorm(size,0,1)
    taus <- cbind(p1p1=Y1_mean(X1,O11,O12,A1=1)+Y2_mean(X1,X2,O11,O12,O23,A2=1),
                  n1p1=Y1_mean(X1,O11,O12,A1=-1)+Y2_mean(X1,X2,O11,O12,O23,A2=1),
                  p1n1=Y1_mean(X1,O11,O12,A1=1)+Y2_mean(X1,X2,O11,O12,O23,A2=-1),
                  n1n1=Y1_mean(X1,O11,O12,A1=-1)+Y2_mean(X1,X2,O11,O12,O23,A2=-1))
    
  }
    d.argmax <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.max)]
    d.argmin <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.min)]
  if(setting == 4){
      df <- data.frame(O,Z.21,Z.22,A1,A2,Y1,Y2,taus,d.argmax,d.argmin)
    }else if(setting %in% c(5,6)){
      df <- data.frame(X1,A1,Y1,X2,A2,Y2,taus,d.argmax,d.argmin)
    }else if(setting %in% c(10:13)){
      df <- data.frame(O11,O12,O23,X1,A1,Y1,X2,A2,Y2,taus,d.argmax,d.argmin)
    }else if(setting == 3){
      df <- data.frame('O1'=O1,A1,Y1,O2.1,A2,Y2,taus,d.argmax,d.argmin)
    }else{
      df <- data.frame('O1'=O1,A1,Y1,O2.1,O2.2,A2,Y2,taus,d.argmax,d.argmin)
    }
    df <- df %>% mutate(d1.star=if_else(substr(d.argmax,5,6)=='+1',1,-1),
                        d2.star=if_else(substr(d.argmax,7,8)=='+1',1,-1)) 
    df <- df[!apply(is.na(df),2,all)]
  return(df)
}
####
# for (s in c(10:13)){
   df10 <- gen_data(size=10,setting=6,seed=1)
#   #mean(apply(df10[,c('p1p1','n1p1','p1n1','n1n1')],1,max))
#   print(mean(apply(df10[,c('Y1','Y2')],1,sum)))
# }
####
gen_disc_data <- function(size,seed,ret='all'){
  set.seed(seed)
  c1 <- 1; c2 <- 1
  # baseline covariates X1,1, . . . , X1,3 are generated according to Bern(p).
  O1.1 <- rbinom(size,1,.5); O1.1[O1.1==0] <- -1
  O1.2 <- rbinom(size,1,.5); O1.2[O1.2==0] <- -1
  O1.3 <- rbinom(size,1,.5); O1.2[O1.3==0] <- -1
  O1 <- cbind(O1.1,O1.2,O1.3)
  # Treatments A1, A2 are randomly generated from {−1, 1} with equal probability 0.5.
  trt <- rbinom(2*size,1,.5); trt[trt==0] <- -1
  A1 <- trt[1:size]; A2 <- trt[(size+1):(2*size)]
  
  # The model for generating outcomes R1 and R2 is defined under the setting stated below:
  
  # Setting 2) 
  # Stage 1 outcome Y1 is generated according to: N((1 +1.5X_13)*A1, 1);
  Y1_mean <- function(O1,A1) expit((O1[,3]-.5*O1[,1])*A1)#expit((1 +1.5*O1[,1]-2.5*O1[,2])*A1)#
  Y1 <- rbinom(size,1,Y1_mean(O1,A1))*c1
  # two intermediate variables, O2,1 ∼ I {N(1.25*X_11*A1, 1) > 0}, and O_22 ∼ I {N(−1.75*O_12*A1, 1) > 0} are generated;
  O2.1 <- as.numeric(1.25*O1[,1]*A1+rnorm(size,0,1)>0); O2.2 <- as.numeric(-1.75*O1[,2]*A1+rnorm(size,0,1)>0)
  # then the Stage 2 outcome Y2 is generated according  to N((0.5 + Y1 + 0.5*A1 +0.5*X_21 − 0.5*X2,2)*A2, 1).
  Y2_mean <- function(O1,O2.2,Y1,A1,A2) expit((.5*O1[,1]+O1[,2]-.2*O2.2+.5*A1+Y1)*A2)#expit((0.5 + Y1  -5*O2.1 - 4.5*O2.2)*A2) #+ 0.5*A1#
  Y2 <- rbinom(size,1,Y2_mean(O1,O2.2,Y1,A1,A2))*c2
  taus <- cbind(p1p1=Y1_mean(O1,A1=1)*c1+Y2_mean(O1,O2.2,Y1,A1=1,A2=1)*c2,
                n1p1=Y1_mean(O1,A1=-1)*c1+Y2_mean(O1,O2.2,Y1,A1=-1,A2=1)*c2,
                p1n1=Y1_mean(O1,A1=1)*c1+Y2_mean(O1,O2.2,Y1,A1=1,A2=-1)*c2,
                n1n1=Y1_mean(O1,A1=-1)*c1+Y2_mean(O1,O2.2,Y1,A1=-1,A2=-1)*c2)
  
  
  d.argmax <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.max)]
  d.argmin <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.min)]
  df <- data.frame(O1,A1,Y1,O2.1,O2.2,A2,Y2,taus,d.argmax,d.argmin)
  df <- df %>% mutate(d1.star=if_else(substr(d.argmax,5,6)=='+1',1,-1),
                      d2.star=if_else(substr(d.argmax,7,8)=='+1',1,-1)) 
  
  if (ret!='all') df <- taus
  return(df)
}

gen_toy_data <- function(size,seed,ret='all'){
  set.seed(seed)
  # Treatments A1, A2 are randomly generated from {−1, 1} with equal probability 0.5.
  trt <- rbinom(2*size,1,.5); trt[trt==0] <- -1
  A1 <- trt[1:size]; A2 <- trt[(size+1):(2*size)]
  
  # The model for generating outcomes R1 and R2 is defined under the setting stated below:
  
  # Setting 2) 
  
  Y1_mean <- function(A1,A2) 1
  Y1 <- 1
  # then the Stage 2 outcome Y2 is generated according  to N((0.5 + Y1 + 0.5*A1 +0.5*X_21 − 0.5*X2,2)*A2, 1).
  Y2_mean <- function(A1,A2) 4*(A1==1 & A2==1)+3*(A1==1 & A2==-1)+5*(A1==-1 & A2==1)+1*(A1==-1 & A2==-1)
  Y2 <- Y2_mean(A1,A2)
  taus <- cbind(p1p1=Y1_mean(A1=1,A2=1)+Y2_mean(A1=1,A2=1),
                n1p1=Y1_mean(A1=-1,A2=1)+Y2_mean(A1=-1,A2=1),
                p1n1=Y1_mean(A1=1,A2=-1)+Y2_mean(A1=1,A2=-1),
                n1n1=Y1_mean(A1=-1,A2=-1)+Y2_mean(A1=-1,A2=-1))
  
  
  d.argmax <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.max)]
  d.argmin <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.min)]
  df <- data.frame(A1,Y1,A2,Y2,taus,d.argmax,d.argmin)
  df <- df %>% mutate(d1.star=if_else(substr(d.argmax,5,6)=='+1',1,-1),
                      d2.star=if_else(substr(d.argmax,7,8)=='+1',1,-1)) 
  
  if (ret!='all') df <- taus
  return(df)
}

# sigmoid surrogate loss functions 
psi1 <- function(x,y) (1+x/(1+abs(x)))*(1+y/(1+abs(y)))
psi2 <- function(x,y) (1+(2/pi)*atan(pi*x/2))*(1+(2/pi)*atan(pi*y/2))
psi3 <- function(x,y) (1+x/sqrt(1+x^2))*(1+y/sqrt(1+y^2))
psi5 <- function(x,y) (1+tanh(x))*(1+tanh(y))
psi4 <- function(x,y) min(x-1,y-1,0)+1

# dat = gen_data(size=1000,setting=6,1)
# initial_theta <- rep(0,2)
# O1.vars <- c('X1')
# O2.bar.vars <- c(O1.vars,'A1','X2')
# cost_C1 <- function(theta,dat,surr_fn,O1.vars,O2.bar.vars){}
#   # assign surrogate function to psi
#   psi <- get(paste('psi',surr_fn,sep=''))
#   # initial parameters
#   theta1 <- theta[1:length(O1.vars)]
#   theta2 <- theta[(length(O1.vars)+1):length(theta)]
#   # stage 1&2 covariates
#   O1 <- dat[,O1.vars]
#   O2.bar <- dat[,O2.bar.vars]
#   # Compute functions f(O;theta) specified in C1
#   dat$f1 <- crossprod(t(as.matrix((O1+1)/2)),theta1)
#   dat$f2 <- crossprod(t(as.matrix(O2.bar)),theta2)
#   
#   
  
#Cost Function (neg. Value fun.)
cost <- function(theta,dat,surr_fn,O1.vars,O2.bar.vars){
  # assign surrogate function to psi
  psi <- get(paste('psi',surr_fn,sep=''))
  theta1 <- theta[1:length(O1.vars)]
  theta2 <- theta[(length(O1.vars)+1):length(theta)]
  # stage 1&2 covariates
  O1 <- dat[,O1.vars]
  O2.bar <- dat[,O2.bar.vars]
  # Compute linear functions f(O,theta)
  dat$f1 <- crossprod(t(as.matrix(O1)),theta1)
  dat$f2 <- crossprod(t(as.matrix(O2.bar)),theta2)
  # Compute value function (missing IPWs)
  V <- with(dat,(Y1+Y2)*psi(A1*f1,A2*f2)) 
  cst <- -sum(V) #+ .5*sum(abs(c(theta1,theta2)))#+ .5*crossprod(c(theta1,theta2))
  return(cst)
}



## Function that fits to the data and predicts
fit.n.predict <- function(df.train,df.test,O1.vars,O2.bar.vars,surr_fn){
  #Intial theta
  initial_theta <- rep(0,length(c(O1.vars,O2.bar.vars)))
  
  #Cost at inital theta
  #cost(initial_theta,dat=df.train,surr_fn=1,O1.vars,O2.bar.vars)
  
  # Derive theta using optim function
  theta_optim <- optim(par=initial_theta,fn=cost,dat=df.train,surr_fn=surr_fn,O1.vars=O1.vars,O2.bar.vars=O2.bar.vars,method="SANN")
  
  #cost at optimal value of the theta
  #theta_optim$value
  
  #set theta
  theta.hat <- theta_optim$par
  
  theta1.hat <- theta.hat[1:length(O1.vars)]
  theta2.hat <- theta.hat[(length(O1.vars)+1):length(theta.hat)]
  # Compute linear functions f(O,theta)
  f1.hat <- crossprod(t(as.matrix(df.test[,O1.vars])),theta1.hat)
  f2.hat <- crossprod(t(as.matrix(df.test[,O2.bar.vars])),theta2.hat)
  #df.test <- data.frame(df.test)
  df.test[[paste('d1.hat.psi',surr_fn,sep='')]] <- as.numeric(sign(f1.hat))
  df.test[[paste('d2.hat.psi',surr_fn,sep='')]] <- as.numeric(sign(f2.hat))
  return(df.test)
}
# df <- gen_data(2500+100000,3,1)
# df.train <- df[1:2500,]
# df.test <- df[2501:100000,]
# surr_fn =  'hinge'
# kernel = 'radial'
# ####
# 
# df.train = read.csv('Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Code/Simulations/df_sepsis_processed.csv')#read.csv('Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Code/Simulations/df_sepsis__.csv')
# df.train <- df.train[sample(1:dim(df.train)[1],2000),]
# df.test = read.csv('Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Code/Simulations/df_test_sepsis__.csv')
# O1.vars <- c('O1_heart_rate','O1_sbp','O1_median_dose_vaso','O1_input_4hourly','O1_input_total_tev')

# O2.vars <- c('O2_heart_rate','O2_sbp','O2_median_dose_vaso','O2_input_4hourly','O2_input_total_tev')
# O2.bar.vars <- c(O1.vars,O2.vars)

####
fit_bowl <- function(df.train,df.test,O1.vars,O2.vars,surr_fn,kernel,lambdas){
  ###################################
  # # features that need no scaling
   non_features <- c('A1','A2','A1.f','A2.f','p1p1','n1p1','p1n1','n1n1','d.argmax','d.argmin','d1.star','d2.star','Y1','Y2')
   non_features <- !(names(df.train) %in% non_features)
   ## find mean and sd column-wise of training data
   trainMean <- apply(df.train[,non_features],2,mean)
   trainSd <- apply(df.train[,non_features],2,sd)
   
   ## center AND scale train & test data using train data stats
   #df.train[,non_features] <- scale(df.train[,non_features])
   #df.test[,non_features] <- sweep(sweep(df.test[,non_features], 2L, trainMean), 2, trainSd, "/")
  
   df.train <- df.train %>% mutate(Y1 = (Y1-min(Y1))/max(Y1-min(Y1)),
                                   Y2 = (Y2-min(Y2))/max(Y2-min(Y2)))
   df.test <- df.test %>% mutate(Y1 = (Y1-min(Y1))/max(Y1-min(Y1)),
                                 Y2 = (Y2-min(Y2))/max(Y2-min(Y2)))
  # print(names(df.test))
  # #
  df.train$Y2 <- df.train$Y2/max(df.train$Y2)
  df.test$Y1 <- df.test$Y1/max(df.test$Y1)
  df.test$Y2 <- df.test$Y2/max(df.test$Y2)
  #
  # # Constant propensity model
   moPropen <- buildModelObj(model = ~1,
                             solver.method = 'glm',
                             solver.args = list('family'='binomial'),
                             predict.method = 'predict.glm',
                             predict.args = list(type='response'))
  
  # # Second stage
   SS.formula <- as.formula(paste('~1+',paste(O2.vars,collapse='+')))
   fitSS <- bowl(sigf = 4,moPropen = moPropen, surrogate = surr_fn,kernel = kernel,#lambdas = 50,
                 data = df.train, reward = df.train$Y2, txName = ifelse('A2' %in% names(df.train),'A2','A2.f'),
                 verbose=0,# cvFolds = 4L,kparam=c(.01,.1,.5,1),
                 regime = SS.formula,kparam=.01)#[1:5]
  # # First stage
   FS.formula <- as.formula(paste('~1+',paste(O1.vars,collapse='+')))
   fitFS <- bowl(sigf = 4,moPropen = moPropen, surrogate=surr_fn,kernel = kernel,#lambdas = 5,
                 data = df.train, reward = df.train$Y1, txName = ifelse('A1' %in% names(df.train),'A1','A1.f'),#[1:3]
                 BOWLObj = fitSS,verbose=0, #cvFolds = 4L,kparam=c(.01,.1,.5,1),
                 regime =FS.formula,kparam=.01)#, lambdas = c(0.5, 1.0), cvFolds = 4L),
  ###################################
  # Estimated value of the optimal treatment regime for training set
  #estimator(fitSS)
  # Estimated optimal treatment for new data
  df.test[[paste('d1.hat.bowl.',surr_fn,sep='')]] <- optTx(fitFS, df.test)[['optimalTx']]
  df.test[[paste('d2.hat.bowl.',surr_fn,sep='')]] <- optTx(fitSS, df.test)[['optimalTx']]
  return(df.test)
}

# df.all <- read.csv('Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Code/Simulations/df_sepsis_processed.csv')
# k_cv = 5
# folds <- sample(1:nrow(df.all)%%k_cv)
# fold <- 1  # which ever fold you want to test with
# df.train <- df.all[folds != fold,]
# df.test <- df.all[folds == fold,]
# 
# fit_bowl(df.train,df.test,O1.vars,O2.vars,surr_fn,kernel,lambdas)


feat_transf <- function(df,setting){
  knots_No <- 2
  #if(is.numeric(setting)){
  if(setting %in% c(1,2,3)){
    features1 <- colnames(df)[grep('O1',colnames(df))]
    features2 <- colnames(df)[grep('O1',colnames(df))]
    
    # Design formula to include intercept, main effects, squares and pairwise interactions
    desing_mat_f1 <- as.formula(paste('~0+(',paste(features1,collapse='+'),')'))
    desing_mat_f2 <- as.formula(paste('~0+(',paste(features2,collapse='+'),')^3'))
    df.splines1 <- data.frame(model.matrix(desing_mat_f1,data=df))
    df.2 <- data.frame(model.matrix(desing_mat_f2,data=df))
    # Natural cubic splines:
    nms1 <- colnames(df.splines1)
    # compute basis for each relevant continuous column
    df.splines1 <- as.data.frame(lapply(nms1, function(nm) {ns(df.splines1[,nm],df = knots_No)}))
    # name the basis
    colnames(df.splines1) <- paste('H1_',1:ncol(df.splines1),sep = '')
    # merge basis with the rest of the columns
    if(setting==1){
      O1.vars <- c(colnames(df.splines1),colnames(df.2),'int')
      O2.bar.vars <- c(O1.vars,'Y1')
    }else{
      O1.vars <- c(colnames(df.splines1),'int')
      O2.bar.vars <- c(O1.vars,colnames(df.2),'Y1','int')
    }
    df <- cbind(df,df.splines1,df.2,'int'=1); df <- df[,unique(colnames(df))]
  }else if (setting=='disc'){
    desing_mat_f <- as.formula(paste('~1+(O1.1+O1.2+O1.3+O2.2+Y1)^5'))
    features_X <- data.frame(model.matrix(desing_mat_f,data=df))
    df <- cbind(df,features_X,'int'=1); df <- df[,unique(colnames(df))]
    O1.vars <- c(colnames(features_X)[-c(grep('O1.2',colnames(features_X)),grep('O2.2',colnames(features_X)),grep('Y1',colnames(features_X)))])
    O2.bar.vars <- c(colnames(features_X)[-grep('O1.3',colnames(features_X))])
  }else if (setting=='toy'){
    O1.vars <- 'A1'
    
    O2.vars <- c('A1','A2')
  }
  return(list(df=df,O1.vars=O1.vars,O2.bar.vars=O2.bar.vars))
}
gen_df <- function(size,setting,sd) {
  # Generate Dataset
  if(is.numeric(setting)){
    df <- gen_data(size,setting,sd)
    }else{
    df <- gen_disc_data(size,sd,ret='all')
    }
  # Combute basis matrix for natural cubic splines
  #df <- feat_transf(df,setting)[['df']]
  return(df)
}
#head(gen_df(size=100,setting=4,sd=1))
run_sims <- function(size,setting,sims_No){
  if(setting==1 |setting==2){df <- gen_data(10,2,1)}else{df <- gen_disc_data(10,1,ret='all')}
  df_ls <- feat_transf(df,setting)
  O1.vars <- df_ls[['O1.vars']]
  O2.bar.vars <- df_ls[['O2.bar.vars']]
  V_fn <- matrix(NA,sims_No,11); errs <- matrix(NA,sims_No,20); time.taken <- matrix(NA,sims_No,10)
  colnames(errs) <- c(paste(rep(paste('d',1:2,'.psi',sep=''),5),rep(c(1:5),each=2),sep=''),
                      paste(rep(paste('d',1:2,'.bowl.',sep=''),4),rep(c('hinge','exp','logit','huber'),each=2),sep=''),'d1.Q','d2.Q');
  
  colnames(V_fn) <- c('True',paste('Estimate.psi',c(1:5),sep=''),paste('Estimate.',c('hinge','exp','logit','huber'),sep=''),'Estimate.Q')
  colnames(time.taken) <- c(paste('psi',c(1:5),sep=''),paste('bowl.',c('hinge','exp','logit','huber'),sep=''),'Q')
  taus_errd1 <- taus_errd2 <- matrix(NA,0,4)
  sd <- sim <- 1
  # Will start loop here:
  while (sim <= sims_No){
    tryCatch({
      cat('Sim No: ',sim,'\n')
      if(setting==1 |setting==2){df <- gen_data(size,setting=setting,sd)}else{df <- gen_disc_data(size,sd,ret='all')}
      # apply(df.train[,c(10,11,12,13)],2,mean)
      df <- feat_transf(df,setting)[['df']]
      df <- df %>% mutate(A1.f = as.factor(A1),A2.f = as.factor(A2))
      df.train <- df[1:as.integer(size/2),]; df.test <- df[1:as.integer(size/2),]; df.test$int <- 1
      ##### fit conc. surrogates
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=1)
      end.time <- Sys.time()
      time.taken[sim,'psi1'] <- as.numeric(end.time - start.time, units = "secs")
      
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=2)
      end.time <- Sys.time()
      time.taken[sim,'psi2'] <- as.numeric(end.time - start.time, units = "secs")
      
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=3)
      end.time <- Sys.time()
      time.taken[sim,'psi3'] <- as.numeric(end.time - start.time, units = "secs")
      
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=4)
      end.time <- Sys.time()
      time.taken[sim,'psi4'] <- as.numeric(end.time - start.time, units = "secs")
      
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=5)
      end.time <- Sys.time()
      time.taken[sim,'psi5'] <- as.numeric(end.time - start.time, units = "secs")
      
      ##### Fit BOWL with surrogates
      start.time <- Sys.time()
      df.test <- fit.bowl(df.train,df.test,O1.vars,O2.bar.vars,surr_fn =  'hinge')#'exp')#
      end.time <- Sys.time()
      time.taken[sim,'bowl.hinge'] <- as.numeric(end.time - start.time, units = "secs")
      ##
      #colnames(df.test) <- gsub('exp','hinge',colnames(df.test))
      ##    
      d1.bowl.hinge.errs <- which(with(df.test,d1.hat.bowl.hinge!=d1.star))
      d2.bowl.hinge.errs <- which(with(df.test,d2.hat.bowl.hinge!=d2.star))
      
      start.time <- Sys.time()
      df.test <- fit.bowl(df.train,df.test,O1.vars,O2.bar.vars,surr_fn = 'exp')
      end.time <- Sys.time()
      time.taken[sim,'bowl.exp'] <- as.numeric(end.time - start.time, units = "secs")
      d1.bowl.exp.errs <- which(with(df.test,d1.hat.bowl.exp!=d1.star))
      d2.bowl.exp.errs <- which(with(df.test,d2.hat.bowl.exp!=d2.star))
      
      start.time <- Sys.time()
      df.test <- fit.bowl(df.train,df.test,O1.vars,O2.bar.vars,surr_fn = 'logit')
      end.time <- Sys.time()
      time.taken[sim,'bowl.logit'] <- as.numeric(end.time - start.time, units = "secs")
      d1.bowl.logit.errs <- which(with(df.test,d1.hat.bowl.logit!=d1.star))
      d2.bowl.logit.errs <- which(with(df.test,d2.hat.bowl.logit!=d2.star))
      
      start.time <- Sys.time()
      df.test <- fit.bowl(df.train,df.test,O1.vars,O2.bar.vars,surr_fn = 'huber')
      end.time <- Sys.time()
      time.taken[sim,'bowl.huber'] <- as.numeric(end.time - start.time, units = "secs")
      d1.bowl.huber.errs <- which(with(df.test,d1.hat.bowl.huber!=d1.star))
      d2.bowl.huber.errs <- which(with(df.test,d2.hat.bowl.huber!=d2.star))
      
      ##### Fit Q-learning
      start.time <- Sys.time()
      ### Second-Stage Analysis
      # outcome model
      if(setting=='disc'){
        moMain <- buildModelObj(model = ~1,
                                solver.method = 'lm')
        moCont <- buildModelObj(model =~O1.1+O1.2+O2.2+A1+Y1,
                                solver.method = 'lm')
      }else if(setting==1){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3+(O1.1+O1.2+O1.3)*A1+Y1,
                                solver.method = 'lm')
        
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3+(O1.1+O1.2+O1.3)*A1+Y1,
                                solver.method = 'lm')
      }else if(setting==2){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3+O2.1+O2.2+(O1.1+O1.2+O1.3+O2.1+O2.2)*A1+Y1,
                                solver.method = 'lm')
        
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3+O2.1+O2.2+(O1.1+O1.2+O1.3+O2.1+O2.2)*A1+Y1,
                                solver.method = 'lm')
      }
      
      # Second stage
      fitSS <- qLearn(moMain = moMain, moCont = moCont,
                      data = df.train, response = df.train$Y2, txName = 'A2')
      
      
      
      ### First-Stage Analysis Main Effects Term
      # main effects model
      if(setting=='disc'){
        moMain <- buildModelObj(model = ~1,
                                solver.method = 'lm')
        moCont <- buildModelObj(model =~O1.3+O1.1,
                                solver.method = 'lm')
      }else if(setting==1 | setting==2){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3,
                                solver.method = 'lm')
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3,
                                solver.method = 'lm')
      }
      fitFS <- qLearn(moMain = moMain, moCont = moCont,
                      data = df.train, response = fitSS, txName = 'A1')
      
      # Estimated value of the optimal treatment regime for training set
      
      # Estimated optimal treatment for new data
      df.test$d1.hat.Q <- optTx(fitFS, df.test)[['optimalTx']]
      df.test$d2.hat.Q <- optTx(fitSS, df.test)[['optimalTx']]
      end.time <- Sys.time()
      time.taken[sim,'Q'] <- as.numeric(end.time - start.time, units = "secs")
      
      d1.Q.errs <- which(with(df.test,d1.hat.Q!=d1.star))
      d2.Q.errs <- which(with(df.test,d2.hat.Q!=d2.star))
      #####
      
      d1.psi1.errs <- which(with(df.test,d1.hat.psi1!=d1.star))
      d2.psi1.errs <- which(with(df.test,d2.hat.psi1!=d2.star))
      d1.psi2.errs <- which(with(df.test,d1.hat.psi2!=d1.star))
      d2.psi2.errs <- which(with(df.test,d2.hat.psi2!=d2.star))
      d1.psi3.errs <- which(with(df.test,d1.hat.psi3!=d1.star))
      d2.psi3.errs <- which(with(df.test,d2.hat.psi3!=d2.star))
      d1.psi4.errs <- which(with(df.test,d1.hat.psi4!=d1.star))
      d2.psi4.errs <- which(with(df.test,d2.hat.psi4!=d2.star))
      d1.psi5.errs <- which(with(df.test,d1.hat.psi5!=d1.star))
      d2.psi5.errs <- which(with(df.test,d2.hat.psi5!=d2.star))
      # Storing taus for which decision rules are wrong
      taus_errd1 <- rbind(taus_errd1,df.test[d1.psi1.errs,c('p1p1','n1p1','p1n1','n1n1')])
      taus_errd2 <- rbind(taus_errd2,df.test[d2.psi1.errs,c('p1p1','n1p1','p1n1','n1n1')])
      # Computing the mean Value with the estimated regimes:
      df.test <- df.test %>% mutate(opt.V=case_when(d1.star==1 & d2.star==1~p1p1,d1.star==1 & d2.star==-1~p1n1,d1.star==-1 & d2.star==1~n1p1,T~n1n1),
                                    V.psi1=case_when(d1.hat.psi1==1 & d2.hat.psi1==1~p1p1,d1.hat.psi1==1 & d2.hat.psi1==-1~p1n1,d1.hat.psi1==-1 & d2.hat.psi1==1~n1p1,T~n1n1),
                                    V.psi2=case_when(d1.hat.psi2==1 & d2.hat.psi2==1~p1p1,d1.hat.psi2==1 & d2.hat.psi2==-1~p1n1,d1.hat.psi2==-1 & d2.hat.psi2==1~n1p1,T~n1n1),
                                    V.psi3=case_when(d1.hat.psi3==1 & d2.hat.psi3==1~p1p1,d1.hat.psi3==1 & d2.hat.psi3==-1~p1n1,d1.hat.psi3==-1 & d2.hat.psi3==1~n1p1,T~n1n1),
                                    V.psi4=case_when(d1.hat.psi4==1 & d2.hat.psi4==1~p1p1,d1.hat.psi4==1 & d2.hat.psi4==-1~p1n1,d1.hat.psi4==-1 & d2.hat.psi4==1~n1p1,T~n1n1),
                                    V.psi5=case_when(d1.hat.psi5==1 & d2.hat.psi5==1~p1p1,d1.hat.psi5==1 & d2.hat.psi5==-1~p1n1,d1.hat.psi5==-1 & d2.hat.psi5==1~n1p1,T~n1n1),
                                    # Q learning and BOWL
                                    V.bowl.hinge=case_when(d1.hat.bowl.hinge==1 & d2.hat.bowl.hinge==1~p1p1,d1.hat.bowl.hinge==1 & d2.hat.bowl.hinge==-1~p1n1,d1.hat.bowl.hinge==-1 & d2.hat.bowl.hinge==1~n1p1,T~n1n1),
                                    V.bowl.exp=case_when(d1.hat.bowl.exp==1 & d2.hat.bowl.exp==1~p1p1,d1.hat.bowl.exp==1 & d2.hat.bowl.exp==-1~p1n1,d1.hat.bowl.exp==-1 & d2.hat.bowl.exp==1~n1p1,T~n1n1),
                                    V.bowl.logit=case_when(d1.hat.bowl.logit==1 & d2.hat.bowl.logit==1~p1p1,d1.hat.bowl.logit==1 & d2.hat.bowl.logit==-1~p1n1,d1.hat.bowl.logit==-1 & d2.hat.bowl.logit==1~n1p1,T~n1n1),
                                    V.bowl.huber=case_when(d1.hat.bowl.huber==1 & d2.hat.bowl.huber==1~p1p1,d1.hat.bowl.huber==1 & d2.hat.bowl.huber==-1~p1n1,d1.hat.bowl.huber==-1 & d2.hat.bowl.huber==1~n1p1,T~n1n1),
                                    
                                    V.Qlearn=case_when(d1.hat.Q==1 & d2.hat.Q==1~p1p1,d1.hat.Q==1 & d2.hat.Q==-1~p1n1,d1.hat.Q==-1 & d2.hat.Q==1~n1p1,T~n1n1))
      V_fn[sim,] <- c(mean(df.test$opt.V),mean(df.test$V.psi1),mean(df.test$V.psi2),mean(df.test$V.psi3),mean(df.test$V.psi4),mean(df.test$V.psi5),
                      mean(df.test$V.bowl.hinge),mean(df.test$V.bowl.exp),mean(df.test$V.bowl.logit),mean(df.test$V.bowl.huber),mean(df.test$V.Qlearn))
      
      errs[sim,c('d1.psi1','d2.psi1')] <- c(length(d1.psi1.errs),length(d2.psi1.errs))/nrow(df.test)
      errs[sim,c('d1.psi2','d2.psi2')] <- c(length(d1.psi2.errs),length(d2.psi2.errs))/nrow(df.test)
      errs[sim,c('d1.psi3','d2.psi3')] <- c(length(d1.psi3.errs),length(d2.psi3.errs))/nrow(df.test)
      errs[sim,c('d1.psi4','d2.psi4')] <- c(length(d1.psi4.errs),length(d2.psi4.errs))/nrow(df.test)
      errs[sim,c('d1.psi5','d2.psi5')] <- c(length(d1.psi5.errs),length(d2.psi5.errs))/nrow(df.test)
      errs[sim,c('d1.bowl.hinge','d2.bowl.hinge')] <- c(length(d1.bowl.hinge.errs),length(d2.bowl.hinge.errs))/nrow(df.test)
      errs[sim,c('d1.bowl.exp','d2.bowl.exp')] <- c(length(d1.bowl.exp.errs),length(d2.bowl.exp.errs))/nrow(df.test)
      errs[sim,c('d1.bowl.logit','d2.bowl.logit')] <- c(length(d1.bowl.logit.errs),length(d2.bowl.logit.errs))/nrow(df.test)
      errs[sim,c('d1.bowl.huber','d2.bowl.huber')] <- c(length(d1.bowl.huber.errs),length(d2.bowl.huber.errs))/nrow(df.test)
      errs[sim,c('d1.Q','d2.Q')] <- c(length(d1.Q.errs),length(d2.Q.errs))/nrow(df.test)
      cat('sim: ',sim,', n: ','setting: ',setting,size,'\n')
      print(round(apply(errs,2,mean,na.rm=T),2))
      print(round(apply(V_fn,2,mean,na.rm=T),2))
      print(round(apply(time.taken,2,mean,na.rm=T),2))
      
      sim <- sim + 1
    }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    sd <- sd + 1
  }
  return(list(errs=errs,V_fn=V_fn,taus_errd1=taus_errd1,taus_errd2=taus_errd2,tot.trials=sd,time.taken=time.taken))
}


run_bowl.Qlearn <- function(size,setting,sims_No){
  if(is.numeric(setting)){
    df <- gen_data(10,setting,1)
    O1.vars <- colnames(df)[grep('O1',colnames(df))]
    O2.bar.vars <- c(colnames(df)[grep('O',colnames(df))],'A1')
  }else{
    df <- gen_disc_data(10,1,ret='all')
    df_ls <- feat_transf(df,setting)
    O1.vars <- df_ls[['O1.vars']]
    O2.bar.vars <- df_ls[['O2.bar.vars']]
    }
  V_fn <- matrix(NA,sims_No,3); errs <- matrix(NA,sims_No,4); time.taken <- matrix(NA,sims_No,2)
  colnames(errs) <- c(paste('d',1:2,'.bowl.hinge',sep=''),'d1.Q','d2.Q');
  
  colnames(V_fn) <- c('True','Estimate.bowl','Estimate.Q')
  colnames(time.taken) <- c('bowl','Q')
  taus_errd1 <- taus_errd2 <- matrix(NA,0,4)
  sd <- sim <- 1
  # Will start loop here:
  while (sim <= sims_No){
    tryCatch({
      cat('Sim No: ',sim,'\n')
      if(is.numeric(setting)){df <- gen_data(size,setting=setting,sd)}else{df <- gen_disc_data(size,sd,ret='all')}
      #df <- feat_transf(df,setting)[['df']]
      df <- df %>% mutate(A1.f = as.factor(A1),A2.f = as.factor(A2))
      df.train <- df[1:as.integer(size/2),]; df.test <- df[1:as.integer(size/2),]; df.test$int <- 1
      ##### Fit BOWL with surrogates
      start.time <- Sys.time()
      df.test <- quiet(fit.bowl(df.train,df.test,O1.vars,O2.bar.vars,surr_fn =  'exp'))#'exp')#
      end.time <- Sys.time()
      time.taken[sim,'bowl'] <- as.numeric(end.time - start.time, units = "secs")
      ##
      d1.bowl.errs <- which(with(df.test,d1.hat.bowl.hinge!=d1.star))
      d2.bowl.errs <- which(with(df.test,d2.hat.bowl.hinge!=d2.star))
      ##### Fit Q-learning
      start.time <- Sys.time()
      ### Second-Stage Analysis
      # outcome model
      if(setting=='disc'){
        moMain <- buildModelObj(model = ~1,
                                solver.method = 'lm')
        moCont <- buildModelObj(model =~O1.1+O1.2+O2.2+A1+Y1,
                                solver.method = 'lm')
      }else if(setting==1){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3+(O1.1+O1.2+O1.3)*A1+Y1,
                                solver.method = 'lm')
        
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3+(O1.1+O1.2+O1.3)*A1+Y1,
                                solver.method = 'lm')
      }else if(setting==2){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3+O2.1+O2.2+(O1.1+O1.2+O1.3+O2.1+O2.2)*A1+Y1,
                                solver.method = 'lm')
        
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3+O2.1+O2.2+(O1.1+O1.2+O1.3+O2.1+O2.2)*A1+Y1,
                                solver.method = 'lm')
      }else if(setting==3){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3+O2.1+(O1.1+O1.2+O1.3+O2.1)*A1+Y1,
                                solver.method = 'lm')
        
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3+O2.1+(O1.1+O1.2+O1.3+O2.1)*A1+Y1,
                                solver.method = 'lm')
      }
      
      # Second stage
      fitSS <- quiet(qLearn(moMain = moMain, moCont = moCont,
                      data = df.train, response = df.train$Y2, txName = 'A2'))
      
      
      
      ### First-Stage Analysis Main Effects Term
      # main effects model
      if(setting=='disc'){
        moMain <- buildModelObj(model = ~1,
                                solver.method = 'lm')
        moCont <- buildModelObj(model =~O1.3+O1.1,
                                solver.method = 'lm')
      }else if(setting==1 | setting==2){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3,
                                solver.method = 'lm')
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3,
                                solver.method = 'lm')
      }
      fitFS <- quiet(qLearn(moMain = moMain, moCont = moCont,
                      data = df.train, response = fitSS, txName = 'A1'))
      
      # Estimated value of the optimal treatment regime for training set
      
      # Estimated optimal treatment for new data
      df.test$d1.hat.Q <- optTx(fitFS, df.test)[['optimalTx']]
      df.test$d2.hat.Q <- optTx(fitSS, df.test)[['optimalTx']]
      end.time <- Sys.time()
      time.taken[sim,'Q'] <- as.numeric(end.time - start.time, units = "secs")
      
      d1.Q.errs <- which(with(df.test,d1.hat.Q!=d1.star))
      d2.Q.errs <- which(with(df.test,d2.hat.Q!=d2.star))
      #####
      
      # Computing the mean Value with the estimated regimes:
      df.test <- df.test %>% mutate(opt.V=case_when(d1.star==1 & d2.star==1~p1p1,d1.star==1 & d2.star==-1~p1n1,d1.star==-1 & d2.star==1~n1p1,T~n1n1),
                                    # Q learning and BOWL
                                    V.bowl.hinge=case_when(d1.hat.bowl.hinge==1 & d2.hat.bowl.hinge==1~p1p1,d1.hat.bowl.hinge==1 & d2.hat.bowl.hinge==-1~p1n1,d1.hat.bowl.hinge==-1 & d2.hat.bowl.hinge==1~n1p1,T~n1n1),
                                    V.Qlearn=case_when(d1.hat.Q==1 & d2.hat.Q==1~p1p1,d1.hat.Q==1 & d2.hat.Q==-1~p1n1,d1.hat.Q==-1 & d2.hat.Q==1~n1p1,T~n1n1))
      V_fn[sim,] <- c(mean(df.test$opt.V),mean(df.test$V.bowl),mean(df.test$V.Qlearn))
      
      errs[sim,c('d1.bowl.hinge','d2.bowl.hinge')] <- c(length(d1.bowl.errs),length(d2.bowl.errs))/nrow(df.test)
      errs[sim,c('d1.Q','d2.Q')] <- c(length(d1.Q.errs),length(d2.Q.errs))/nrow(df.test)
      cat('sim: ',sim,', n: ','setting: ',setting,size,'\n')
      print(round(apply(errs,2,mean,na.rm=T),2))
      print(round(apply(V_fn,2,mean,na.rm=T),2))
      print(round(apply(time.taken,2,mean,na.rm=T),2))
      sim <- sim + 1
    }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    sd <- sd + 1
  }
  write.csv(x = cbind('d1_err'=errs[,1],'d2_err'=errs[,2],'time'=time.taken[,1],'V_of_DTR_hat'=V_fn[,2],'V_of_DTR_star'=V_fn[,1]),
            file = paste('../Results/BOWL_size_',as.integer(size/2),'_setting_',setting,'_sims_No_',sims_No,'.csv',sep=''))
  
  write.csv(x = cbind('d1_err'=errs[,3],'d2_err'=errs[,4],'time'=time.taken[,2],'V_of_DTR_hat'=V_fn[,3],'V_of_DTR_star'=V_fn[,1]),
            file = paste('../Results/Qlearning_size_',as.integer(size/2),'_setting_',setting,'_sims_No_',sims_No,'.csv',sep=''))
  return(list(errs=errs,V_fn=V_fn,taus_errd1=taus_errd1,taus_errd2=taus_errd2,tot.trials=sd,time.taken=time.taken))
}


#####
# library(ggplot2)
# library(latex2exp)
# b <- seq(-3 ,3, by=0.1) # sin/cos are periodic, no point going past 2*pi
# circ <- data.frame(x=b, y=b^2)
# dat=data.frame(x=c(0.1,1.5,-1.5,0.1),y=c(0.7,0.5,0.5,-.5),sign=c('+','-','-','-'))
# ggplot(dat, aes(x, y)) +
#   geom_text(aes(label=sign,color=sign),size = 10) +
#   coord_cartesian(xlim=c(-2, 2), ylim=c(-1, 1)) +
#   geom_polygon(dat=circ,alpha=.3,fill='black') +
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_blank(),
#         axis.title.x = element_text(size = 16),
#         axis.title.y = element_text(size = 16)) +
#   geom_vline(xintercept = 0, colour = "black") +
#   geom_hline(yintercept = 0, colour = "black") +
#   xlab(TeX('$\\X_{1}$')) +
#   ylab(TeX('$\\X_{2}$')) 
# 
# 
# 
# df <- data.frame(x=c(-3,-1,
#                      -1,1,
#                      1,3),y=c(-1,-1,1,1,-1,-1),grp=c(1,1,2,2,3,3))
# ggplot(df, aes(x, y,group=grp)) +
#   geom_line(size=2) +
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#       panel.background = element_blank(), axis.line = element_blank(),
#       axis.title.x = element_text(size = 16),
#       axis.title.y = element_text(size = 16)) +
#   geom_vline(xintercept = 0, colour = "black") +
#   geom_hline(yintercept = 0, colour = "black") +
#   xlab(TeX('$\\X_{1}$')) +
#   ylab(TeX('$\\mu^*_{1}(X_{1})$')) 
#   
# 
