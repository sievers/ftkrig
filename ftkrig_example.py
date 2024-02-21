import numpy as np
import ftkrig
from scipy import interpolate
from matplotlib import pyplot as plt
plt.ion()


N=1025  #using an odd number here since the highest-k mode loses power for even-length FFTs

x=np.arange(N)

NN=N//2+1
ps=np.ones(NN)
alpha=-1.0
ps=(1.0+np.arange(NN))**alpha
y=np.random.randn(N)
yft=np.fft.rfft(y)
y=np.fft.irfft(yft*np.sqrt(ps),n=N)
y=y/np.sqrt(np.mean(ps)) #get us back to unit expected variance in the data

nfine=64  #how finely divided we want the interval between grid points
npal=20    #half-width of data window we'll use to interpolate

coeffs,evec,einterp=ftkrig.get_coeffs_1d(ps,nfine,npal,N,do_errs=True,renorm=False)
print('max typical fractional error on fine grid: ',np.sqrt(evec.max()))
print('max typical fractional error from linear interpolation on fine grid: ',np.sqrt(einterp))

xx=np.random.rand(N)+x
yy=ftkrig.ftinterp_exact(y,xx)
yy_krig=ftkrig.eval_krig(xx,y,coeffs)
print('observed kriging interpolation error: ',np.std(yy-yy_krig))
print('RMS values of true, Kriged values: ',np.std(yy),np.std(yy_krig))



#do a spline, wrap around so there no edge effects where we're interpolating
spln=interpolate.splrep(np.hstack([x-N,x,x+N]),np.hstack([y,y,y]))
yy_spline=interpolate.splev(xx,spln)
print('spline error: ',np.std(yy-yy_spline))
print('RMS of spline values: ',np.std(yy_spline))

plt.clf()
plt.plot(xx,yy,'*')
plt.plot(xx,yy_spline,'+')
plt.plot(xx,yy_krig,'.')
#plt.plot(yy,yy-yy_spline,'*')
#plt.plot(yy,yy-yy_krig,'.')

