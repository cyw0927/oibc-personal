"""Full nonseparable shrinkage covariance inverse, exact via Woodbury."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize,check_grad

class FullCovariance:
 def __init__(self,samples,missing=None,peer_groups=None,min_shrink=.01):
  samples=np.asarray(samples,float)
  if samples.ndim!=2 or len(samples)<3 or not np.isfinite(samples).all():raise ValueError('invalid covariance samples')
  n,p=samples.shape; centered=samples-samples.mean(axis=0);var=(centered*centered).mean(axis=0);positive=var[var>1e-16]
  if len(positive)==0:raise ValueError('no variance')
  self.variance_floor=max(float(np.median(positive))*1e-4,1e-12);var=np.maximum(var,self.variance_floor)
  if missing:
   if peer_groups is None or p%60:raise ValueError('missing site needs peer metadata')
   for idx in missing:
    peers=[j for j in range(len(peer_groups)) if j not in missing and peer_groups[j]==peer_groups[idx]]
    for level in range(60):
     var[idx*60+level]=max(float(np.median(var[np.array(peers)*60+level])),self.variance_floor)
  standard=centered/np.sqrt(var);d=(standard*standard).mean(axis=0);gram=standard@standard.T
  off=max(float(np.sum(gram*gram)/(n*n)-np.sum(d*d)),0.)
  squares=standard*standard;num=float((np.sum(np.sum(squares,axis=1)**2)-np.sum(squares*squares)-n*off)/(n*(n-1)))
  estimated=1. if off<=1e-30 else float(np.clip(num/off,0,1));self.estimated_shrinkage=estimated;self.shrinkage=max(estimated,min_shrink)
  self.diagonal=self.shrinkage*var+self.variance_floor*1e-6
  self.u=centered*np.sqrt((1-self.shrinkage)/n)
  self.du=self.u/self.diagonal
  k=np.eye(n)+self.du@self.u.T;self.factor=cho_factor(k,lower=True,check_finite=True)
  self.n=n;self.p=p;self.rank=int(np.linalg.matrix_rank(centered));self.missing_count=len(missing or [])
 def solve(self,vector):
  v=np.asarray(vector,float).reshape(-1); dv=v/self.diagonal
  return dv-self.du.T@cho_solve(self.factor,self.u@dv,check_finite=False)

def joint_objective(x,shape,A,T,Y,cov):
 z=x.reshape(shape);err=A@z@T.T-Y;q=cov.solve(err.ravel()).reshape(err.shape)
 return float(np.sum(err*q)),(2*A.T@q@T).ravel()

def synthetic_check():
 rng=np.random.default_rng(873);samples=rng.normal(size=(12,9));cov=FullCovariance(samples)
 explicit=np.diag(cov.diagonal)+cov.u.T@cov.u;v=rng.normal(size=9);sol=cov.solve(v);exact=np.linalg.solve(explicit,v);reldiff=float(np.linalg.norm(sol-exact)/np.linalg.norm(exact))
 A=np.array([[1.,0.],[0.,1.],[.3,.7]]);T=np.array([[1.,0.],[0.,1.],[.5,.5]]);Y=rng.random((3,3));f=lambda x:joint_objective(x,(2,2),A,T,Y,cov)
 x=rng.random(4);gerr=float(check_grad(lambda z:f(z)[0],lambda z:f(z)[1],x));opts={'ftol':1e-14,'gtol':1e-10,'maxiter':5000}
 rr=[minimize(f,z,jac=True,bounds=[(0,None)]*4,method='L-BFGS-B',options=opts) for z in [np.zeros(4),np.ones(4)*4]];diff=abs(rr[0].fun-rr[1].fun)
 if reldiff>1e-9 or gerr>1e-5 or diff>1e-8 or not all(r.success for r in rr):raise ValueError('synthetic full covariance verification failed')
 return {'woodbury_relative_error':reldiff,'gradient_error':gerr,'two_start_objective_diff':float(diff)}
