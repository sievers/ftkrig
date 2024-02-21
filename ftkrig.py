import numpy as np
import numba as nb
from scipy.linalg import toeplitz

def gen_corr(x,ps,N=None):
    if N is None:
        N=2*(len(ps)-1)
    corr=0*x+ps[0]
    if N%2==0:
        for k in np.arange(1,len(ps)-1):
            corr=corr+2*np.cos(2*np.pi*x*k/N)*ps[k]
        corr=corr+np.cos(np.pi*x)
    else:
        for k in np.arange(1,len(ps)):
            corr=corr+2*np.cos(2*np.pi*x*k/N)*ps[k]
    return corr


def ftinterp_exact(dat,x):
    datft=np.fft.rfft(dat)
    tot=np.real(datft[0])
    N=len(dat)
    if N%2==1:
        for k in range(1,len(datft)):
            tot=tot+2*np.real(datft[k]*np.exp(2J*np.pi*x*k/N))
    else:
        for k in range(1,len(datft)-1):
            tot=tot+2*np.real(datft[k]*np.exp(2J*np.pi*x*k/N))
        tot=tot+np.cos(np.pi*x)*np.real(datft[-1])
    return tot/N
    
def get_coeffs_1d(ps,nfine,n,N=None,do_errs=False,renorm=False,eigthresh=0):
    if N is None:
        N=2*(len(ps)-1)

        
    xgrid=np.arange(0,2*n+1)
    corrgrid=gen_corr(xgrid,ps,N) #get correlation function on our coarse grid
    coeffs=np.zeros([nfine+1,len(corrgrid)])
    xx=np.arange(-n,n+1)
    mat=toeplitz(corrgrid) #this is now the coarse grid correlation matrix

    #if we use a fat window with a smooth signal, we are
    #overdetermined and need to do something like a pinv
    if eigthresh>0:
        minv=np.linalg.pinv(mat,eigthresh)
    else:
        minv=np.linalg.inv(mat)
    if do_errs:
        evec=np.zeros(nfine+1)
    #now loop over a fine grid (from 0 to 1) to pre-calculate
    #the kriging weights
    for i in range(nfine+1):
        myx=xx-(1.0*i/nfine)  #offset of our desired point from the grid points
        cvec=gen_corr(myx,ps,N)
        coeffs[i,:]=cvec@minv
        if renorm:  #set this to true if you want the mean/variance preserved
            myvar=coeffs[i,:]@mat@coeffs[i,:]/mat[0,0]
            coeffs[i,:]=coeffs[i,:]/np.sqrt(myvar)
            coeffs[i,:]=coeffs[i,:]-np.mean(coeffs[i,:])+1/(coeffs.shape[1])
            myvar=coeffs[i,:]@mat@coeffs[i,:]/mat[0,0]
            coeffs[i,:]=coeffs[i,:]/np.sqrt(myvar)

        if do_errs:
            evec[i]=(mat[0,0]-cvec@minv@cvec)  #from partitioned inverse to get bottom-right element
    if do_errs:
        #since we're lazy, we're going to linearly interpolate between fine grid points
        #to get to our arbitrary point.  Find what that expected error is the same way
        #we did before, but this time, we have two points at adjacent fine grid locations
        #and we're going to look at the error halfway between them.
        myx=np.asarray([0,1,0.5])/nfine
        cvec=gen_corr(myx,ps,N)
        mat=toeplitz(cvec)
        minv=np.linalg.inv(mat)
        einterp=1/minv[-1,-1]
        evec[evec<0]=0 #this should be non-negative, but well roundoff happens
    if do_errs:
        #normalize to fractional errors vs. signal variance
        psnorm=2*np.sum(ps)
        return coeffs,evec/psnorm,einterp/psnorm
    else:
        return coeffs
    
@nb.njit(parallel=False)
def eval_krig(x,dat,coeffs):
    npal=coeffs.shape[1]//2
    nfine=coeffs.shape[0]-1
    n=len(x)
    out=np.empty(n)
    for i in nb.prange(n):
        xi=int(x[i])
        frac=(x[i]-xi)*nfine
        ci=int(frac)
        ifrac=frac-ci
        sum1=0
        sum2=0
        if xi<n-npal-1:
            for j in np.arange(xi-npal,xi+npal+1):
                sum1=sum1+coeffs[ci,j-xi+npal]*dat[j]
                sum2=sum2+coeffs[ci+1,j-xi+npal]*dat[j]
        else:
            for j in np.arange(xi-npal,xi+npal+1):
                sum1=sum1+coeffs[ci,j-xi+npal]*dat[j%n]
                sum2=sum2+coeffs[ci+1,j-xi+npal]*dat[j%n]

        out[i]=sum2*ifrac+sum1*(1-ifrac)
        if i==100:
            print(xi,frac,ci,ifrac,sum1,sum2,out[i])
            
    return out
    
