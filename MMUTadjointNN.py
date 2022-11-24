from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, lax, vmap, grad, jacobian, random, pmap, soft_pmap

import numpy as np
from read_traj import *

import matplotlib
import matplotlib.pyplot as plt

import scipy.optimize
import optax
import time
import haiku as hk

mol = 'heh+'
basis = 'sto-3g'
init = 'hf'
td = 'rt-tdexx'

dt = 0.08268

inpath = './datafiles'
inpath2 = './mydata/'
outpath = './'
saved_model_dir = './models/'

# 0 or 1, whether to propagate training trajectories or not
training_prop = 1 #int(float(sys.argv[1]))

# no. of training pts used per trajectory to train the loaded model
ntrainpt = 18000 #int(float(sys.argv[2]))

fieldfreq = 1.0842
fieldfreq_ndyaglaser = 0.0428
flddct = {
    'delta':[np.inf,0.0],
    'ress0s11cyc':[fieldfreq,(2*np.pi/fieldfreq)],
    'ress0s1HOMOminocc5cyc':[fieldfreq,345.201],
    'ress0s1HOMOminocc':[fieldfreq,69.0401],
    'ress0s1HOMOhalfminocc':[fieldfreq,41.3414],
    'doubress0s11cyc':[2*fieldfreq,(np.pi/fieldfreq)],
    'halfress0s11cyc':[0.5*fieldfreq,(4*np.pi/fieldfreq)],
    'ndlaser1cyc':[fieldfreq_ndyaglaser,(2*np.pi/fieldfreq_ndyaglaser)]
}
amplst = [0.005, 0.05, 0.5]

a = list(flddct.keys())        

allfld, allfrq, allamp, alltme, norm_direc = [], [], [], [], []
nalltraj = 100
for i in range(nalltraj):
    ampj = 0.05
    allfld.append(a[0]+'='+str(ampj))
    thisfreq = fieldfreq*(i+1)*2/100 
    allfrq.append(thisfreq)
    allamp.append(ampj)
    alltme.append(10*np.pi/thisfreq)
    norm_direc.append(np.array([0.0,0.0,1.0]))

offset = 0
mmut_freq = 100
cleanup = True
tprop = 18000

mynumsteps = 4000
ntvec = mynumsteps

td_name = allfld[0]
try:
    for i in allfld[1:]:
        td_name += '+' + i
except:
    pass

normalization = 1 # (ntrain-2)*drc**2

ii = 0
allden = []
for i in allfld:
    if ii==0:
        print(f'reading DOFs and AO matrices from {i}')
        a = traj_data(inpath, mol, basis, init, td, i,ii,  npz_format=True)
        a.assign_init_data()
        drc = a.drc
        xmat, didat, kinmat, enmat, eeten = a.xmat, a.didat, a.kinmat, a.enmat, a.eeten
        # pdot, xinp, yinp = np.zeros((0,drc,drc)), np.zeros((0,drc,drc)), np.zeros((0,drc,drc))
        # hfldinp_o = np.zeros((0,drc,drc))
    print(f'reading training TD data from {i}')
    a = traj_data(inpath, mol, basis, init, td, i, ii, npz_format=True)
    a.assign_init_data()
    a.assign_td_data(dt, clean_step=mmut_freq, cleanup=cleanup)
    _, _, _, _, p_all_oi = a.clip_our_td_data(ourpath=inpath2, offset=offset, tt=tprop, ntrain=ntrainpt)
   # _, _, _, _, p_all_oi = a.clip_td_data(offset=offset, tt=tprop, ntrain=ntrainpt)
    allden.append(p_all_oi)
    ii += 1

m = drc

# pars_file = saved_model_dir + "savedparams.npz"
# thetastar = np.load(pars_file, allow_pickle=True)['arr_0']
# thetastar = 1e-2 * np.random.normal(size=2*drc**4)
# beta1 = thetastar[ : drc**4].reshape((drc**2, drc**2))
# gamma1 = thetastar[drc**4 : 2*drc**4].reshape((drc**2, drc**2))

# need two masks
# upper mask is matrix whose (u,v)-th element is 0 unless u <= v
# lower mask is matrix whose (u,v)-th element is 0 unless u > v
upper = np.zeros((2,2),dtype=np.float64)
lower = np.zeros((2,2),dtype=np.float64)
for u in range(2):
    for v in range(2):
        if u <= v:
            upper[u,v] = 1.0
        if u > v:
            lower[u,v] = 1.0

X = xmat
ru1 = 2*np.einsum('uv,uk,ma,sb,uvms,vl->klab',upper,X,X,X,a.eeten,X)
ru2 = np.einsum('uv,uk,ma,sb,umvs,vl->klab',upper,X,X,X,a.eeten,X)
rl1 = 2*np.einsum('uv,uk,ma,sb,vums,vl->klab',lower,X,X,X,a.eeten,X)
rl2 = np.einsum('uv,uk,ma,sb,vmus,vl->klab',lower,X,X,X,a.eeten,X)
beta1trueNP = ru1 - ru2 + rl1 - rl2
gamma1trueNP = ru1 - ru2 - rl1 + rl2

beta0trueNP = xmat.conj().T @ (kinmat - enmat) @ xmat
gamma0trueNP = np.zeros(beta0trueNP.shape)

# extract parameters for real target
beta0true = -beta0trueNP.reshape((-1))
beta1true = -beta1trueNP.reshape((drc**2, drc**2))
# extract parameters for imag target
gamma0true = -gamma0trueNP.reshape((-1))
gamma1true = -gamma1trueNP.reshape((drc**2, drc**2))

beta0 = beta0true
gamma0 = gamma0true

thetatrue = jnp.concatenate([beta1true.reshape((-1)), gamma1true.reshape((-1))])

# cmdbeta0 = np.load('cmdbeta0.npz')['beta0']
# cmdbeta1 = np.load('cmdbeta1.npz')['beta1']
# cmdgamma0 = np.load('cmdgamma0.npz')['gamma0']
# cmdgamma1 = np.load('cmdgamma1.npz')['gamma1']

# print("CHECKING EXACT BETAs and GAMMAs:")
# print(np.linalg.norm(cmdbeta0 - beta0true))
# print(np.linalg.norm(cmdbeta1 - beta1true))
# print(np.linalg.norm(cmdgamma0 - gamma0true))
# print(np.linalg.norm(cmdgamma1 - gamma1true))
# print("END CHECK")



#============================================================================================================#
def flattener(td):
    alldata = []
    allkeys = td.keys()
    for key in allkeys:
        subkeys = td[key].keys()
        for subkey in subkeys:
            alldata.append(td[key][subkey].reshape((-1)))
    
    return jnp.concatenate(alldata)

def populator(td, fl):
    newtd = {}
    allkeys = td.keys()
    cnt = 0
    for key in allkeys:
        subkeys = td[key].keys()
        for subkey in subkeys:
            if key not in newtd:
                newtd[key] = {}
            numelts = td[key][subkey].reshape((-1)).shape[0]
            newtd[key][subkey] = fl[cnt:(cnt+numelts)].reshape(td[key][subkey].shape)
            cnt += numelts
    return newtd

def gradflattener(td):
    alldata = []
    allkeys = td.keys()
    for key in allkeys:
        subkeys = td[key].keys()
        for subkey in subkeys:
            alldata.append(td[key][subkey].reshape((drc**2, -1)))
    
    return jnp.concatenate(alldata, axis=1)
#============================================================================================================#

# assume p is of size drc x drc    #
#latter parameters are for forcing, aka everything past p
def MLhamNN(x,y): #t, fldfrq,fldamp,tmeoff,norm_direc):
    layerwidth = 256
    # inpmod = hk.Linear(layerwidth,w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"))
    # hmod = hk.Linear(layerwidth,w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"))
    # outmod = hk.Linear(2*drc**2,w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"))
    mlp = hk.Sequential([ 
        hk.Linear(layerwidth,1,w_init =None ),
        jax.nn.selu,
        hk.Linear(layerwidth,1, w_init = None),
        jax.nn.selu,
        hk.Linear(layerwidth,1,w_init = None),
        jax.nn.selu,
        hk.Linear(2*drc**2, w_init = None)])
    x = x.flatten()
    y = y.flatten()
    inplyr = jnp.array(jnp.concatenate([x,y]))
    h = mlp(inplyr)
    hreal = h[0:drc**2]
    himag = h[drc**2:]
    symmetrized_real = 0.5*(hreal.reshape((drc,drc)) + hreal.reshape((drc,drc)).T) 
    antisymmetrized_imag = 0.5*(himag.reshape((drc,drc)) - himag.reshape((drc,drc)).T)
    return symmetrized_real, antisymmetrized_imag


MLham = hk.transform(MLhamNN)
MLham = hk.without_apply_rng(MLham)
rng = random.PRNGKey(42)
params = MLham.init(rng, allden[0][0,:,:].reshape((-1)).real,allden[0][0,:,:].reshape((-1)).imag)
numparams = len(flattener(params))
print('numparams: ', numparams)

mydHdtheta = jacobian(MLham.apply, 0)
mydHdX = jacobian(MLham.apply, 1)
mydHdY = jacobian(MLham.apply,2)

def expderiv(d, u, w):
    offdiagmask = jnp.ones((m, m)) - jnp.eye(m)
    expspec = jnp.exp(d)
    e1, e2 = jnp.meshgrid(expspec, expspec)
    s1, s2 = jnp.meshgrid(d, d)
    denom = offdiagmask * (s1 - s2) + jnp.eye(m)
    mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
    prederivamat = jnp.einsum('ij,jk,kl->il',u.conj().T,w,u) 
    derivamat = prederivamat * mask
    return jnp.einsum('ij,jk,kl->il',u,derivamat,u.conj().T)

def expderiv2(d, u, w):
    offdiagmask = jnp.ones((m, m)) - jnp.eye(m)
    expspec = jnp.exp(d)
    e1, e2 = jnp.meshgrid(expspec, expspec)
    s1, s2 = jnp.meshgrid(d, d)
    denom = offdiagmask * (s1 - s2) + jnp.eye(m)
    mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
    prederivamat = jnp.einsum('ij,abjk,kl->ilab',u.conj().T,w,u) 
    derivamat = jnp.einsum('ilab,il->ilab',prederivamat,mask)
    return jnp.einsum('ij,jkab,kl->ilab',u,derivamat,u.conj().T)

def expderiv3(d, u, w):
    offdiagmask = jnp.ones((m, m)) - jnp.eye(m)
    expspec = jnp.exp(d)
    e1, e2 = jnp.meshgrid(expspec, expspec)
    s1, s2 = jnp.meshgrid(d, d)
    denom = offdiagmask * (s1 - s2) + jnp.eye(m)
    mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
    #weights = jnp.concatenate([gradflattener(w[0]), gradflattener(w[1])]).T
    #modified einsum here
    prederivamat = jnp.einsum('ij,ajk,kl->ila',u.conj().T,gradflattener(w).reshape(numparams,drc,drc),u) 
    #prederivamat = jnp.einsum('ij,abcdjk,kl->ilabcd',u.conj().T,w,u) 
    #modified einsum here
    derivamat = jnp.einsum('ila,il->ila',prederivamat,mask)
    #modified einsum here
    return jnp.einsum('ij,jka,kl->ila',u,derivamat,u.conj().T)

def xicomp(hkparams, x, y, evals, evecs):
    jacR = mydHdX(hkparams, x, y)
    jacI =  mydHdY(hkparams, x, y)
    dHdp = 0.5 * ((jacR[0] + 1j*jacR[1]) - 1j*(jacI[0] +1j*jacI[1]))
    dHdPbar = 0.5 * ((jacR[0] + 1j*jacR[1]) + 1j*(jacI[0] +1j*jacI[1]))
    dHdp = dHdp.reshape((drc,drc,drc,drc))
    dHdPbar = dHdPbar.reshape((drc,drc,drc,drc))
    jacP = expderiv2(evals, evecs, dHdp)
    jacPbar = expderiv2(evals, evecs, dHdPbar)
    return jacP, jacPbar

def dUdtheta(hkparams, x, y, evals, evecs):
    # x = jnp.real(p)
    # y = jnp.imag(p)

    # beta1 = theta[:drc**4].reshape((drc**2, drc**2))
    # gamma1 = theta[drc**4:].reshape((drc**2, drc**2))
    # dHdtheta = 0.5*jnp.einsum('ab,ck,dl->abcdkl',x.reshape((drc,drc)),jnp.eye(drc),jnp.eye(drc))
    # dHdtheta += 0.5*jnp.einsum('ab,cl,dk->abcdkl',x.reshape((drc,drc)),jnp.eye(drc),jnp.eye(drc))
    
#     jacbeta = expderiv3(evals, evecs, dHdtheta)
    
#     dHdtheta = 0.5j*jnp.einsum('ab,ck,dl->abcdkl',y.reshape((drc,drc)),jnp.eye(drc),jnp.eye(drc))
#     dHdtheta -= 0.5j*jnp.einsum('ab,cl,dk->abcdkl',y.reshape((drc,drc)),jnp.eye(drc),jnp.eye(drc))
#     jacgamma = expderiv3(evals, evecs, dHdtheta)
    
#     tmp = [jacbeta.reshape((drc,drc,drc**4)), jacgamma.reshape((drc,drc,drc**4))]
#     return jnp.concatenate(tmp, axis=2)
    dHdtheta =  mydHdtheta(hkparams, x, y)
   # import pdb; pdb.set_trace()
    tm1 = expderiv3(evals, evecs,dHdtheta[0])
    tm2 = expderiv3(evals, evecs,dHdtheta[1])
    return tm1+tm2

def adjgrad(hkparams, Ptilde, tmeoff, fldfrq, fldamp, norm_direc):
    tvec = dt*jnp.arange(ntvec)
    P0 = Ptilde[0,:,:]
    propagated_dens = [P0]
    H0 = MLham.apply(hkparams,P0.real,P0.imag)
    H0 = H0[0] + 1j*H0[1]
    evals, evecs = jnp.linalg.eigh(H0)
    U0 = evecs @ jnp.diag(jnp.exp(-1j*dt*evals)) @ evecs.conj().T
    P1 = U0 @ P0 @ U0.conj().T
    propagated_dens.append( P1 )
    # 
    def bodyfun(i, dtup):
        dl, dvals, dvecs, dU = dtup
        P0 = dl[i, :, :]
        P1 = dl[i+1, :, :]
        H1 = MLham.apply(hkparams,P1.real,P1.imag)
        H1 = H1[0] + 1j*H1[1]
        evals, evecs = jnp.linalg.eigh(H1)
        dvals = dvals.at[i+1].set( evals )
        dvecs = dvecs.at[i+1].set( evecs )
        U1 = evecs @ jnp.diag(jnp.exp(-2j*dt*evals)) @ evecs.conj().T
        dU = dU.at[i+1].set( U1 )
        P2 = U1 @ P0 @ U1.conj().T
        dl = dl.at[i+2].set( P2 )
        return (dl, dvals, dvecs, dU)
    # 
    alldens = jnp.concatenate([jnp.stack(propagated_dens), jnp.zeros((ntvec-1, drc, drc))], axis=0)
    allevals = jnp.concatenate([jnp.expand_dims(evals,0), jnp.zeros((ntvec-1, drc))], axis=0)
    allevecs = jnp.concatenate([jnp.expand_dims(evecs,0), jnp.zeros((ntvec-1, drc, drc))], axis=0)
    allU = jnp.concatenate([jnp.expand_dims(U0,0), jnp.zeros((ntvec-1, drc, drc))], axis=0)
    ftup = lax.fori_loop(0, ntvec-1, bodyfun, (alldens, allevals, allevecs, allU))
    Pstack, allevals, allevecs, allU = ftup
    # 
    def bodylamb(i, dl):
        k = ntvec - i
        newlamb = Pstack[k, :, :] - Ptilde[k, :, :]
        xiak = xicomp(hkparams, Pstack[k].reshape((-1)).real, Pstack[k].reshape((-1)).imag,-2j*dt*allevals[k], allevecs[k])
        xibk = xicomp(hkparams, Pstack[k].reshape((-1)).real, Pstack[k].reshape((-1)).imag, 2j*dt*allevals[k], allevecs[k])
        # if i <---> k correspond, then i-1 <---> k+1 correspond
        newlamb += jnp.einsum('ij,ijkl->kl',dl[i-1] @ allU[k] @ Pstack[k-1], (-2j*dt*xiak[0]).conj())
        newlamb += jnp.einsum('ij,ijkl->kl',dl[i-1] @ allU[k] @ Pstack[k-1], (-2j*dt*xiak[1]).conj()).conj()
        newlamb += jnp.einsum('ij,ijkl->kl',Pstack[k-1] @ allU[k].conj().T @ dl[i-1], (2j*dt*xibk[0]).conj())
        newlamb += jnp.einsum('ij,ijkl->kl',Pstack[k-1] @ allU[k].conj().T @ dl[i-1], (2j*dt*xibk[1]).conj()).conj()
        newlamb += allU[k+1].conj().T @ dl[i-2] @ allU[k+1]
        return dl.at[i].set( newlamb )
    # 
    lambfinal = Pstack[ntvec, :, :] - Ptilde[ntvec, :, :]
    lambnext = Pstack[ntvec-1, :, :] - Ptilde[ntvec-1, :, :]
    xiak = xicomp(hkparams, Pstack[ntvec-1].reshape((-1)).real, Pstack[ntvec-1].reshape((-1)).imag, -2j*dt*allevals[ntvec-1], allevecs[ntvec-1])
    xibk = xicomp(hkparams, Pstack[ntvec-1].reshape((-1)).real, Pstack[ntvec-1].reshape((-1)).imag,  2j*dt*allevals[ntvec-1], allevecs[ntvec-1])
    lambnext += jnp.einsum('ij,ijkl->kl',lambfinal @ allU[ntvec-1] @ Pstack[ntvec-2], (-2j*dt*xiak[0]).conj())
    lambnext += jnp.einsum('ij,ijkl->kl',lambfinal @ allU[ntvec-1] @ Pstack[ntvec-2], (-2j*dt*xiak[1]).conj()).conj()
    lambnext += jnp.einsum('ij,ijkl->kl',Pstack[ntvec-2] @ allU[ntvec-1].conj().T @ lambfinal, (2j*dt*xibk[0]).conj())
    lambnext += jnp.einsum('ij,ijkl->kl',Pstack[ntvec-2] @ allU[ntvec-1].conj().T @ lambfinal, (2j*dt*xibk[1]).conj()).conj()
    lambcat = jnp.concatenate([jnp.expand_dims(lambfinal,0), jnp.expand_dims(lambnext, 0), jnp.zeros((ntvec-2, drc, drc))])
    alllamb = lax.fori_loop(2, ntvec, bodylamb, lambcat)
    lambstack = jnp.flipud( jnp.stack(alllamb, axis=0) )    
    # 
    def bodygrad(k, gL):
        tmp1 = -2j*dt*(dUdtheta(hkparams, Pstack[k].real, Pstack[k].imag,-2j*dt*allevals[k], allevecs[k]))
        term1 = jnp.einsum('ija,jk,kl->ila',tmp1,Pstack[k-1],allU[k].conj().T)
        term2 = term1.transpose((1,0,2)).conj()
        return gL + jnp.real(jnp.einsum('il,ila->a',lambstack[k],(term1+term2).conj()))
    # 
    tmp1 = -1j*dt*(dUdtheta(hkparams, Pstack[0].real,  Pstack[0].imag, 1j*dt*allevals[0], allevecs[0]))
    term1 = jnp.einsum('ija,jk,kl->ila',tmp1,Pstack[0],allU[0].conj().T)
    term2 = term1.transpose((1,0,2)).conj()
    initgradL = jnp.real(jnp.einsum('il,ila->a',lambstack[0],(term1+term2).conj()))
    gradL = lax.fori_loop(1, ntvec, bodygrad, initgradL)/ntvec
    return gradL
    
def MMUT_Prop_HSB(hkparams, initial_density, tmeoff=1, fldfrq=1, fldamp=1, norm_direc=jnp.array([0.0,0.0,1.0])):
    tvec = dt*jnp.arange(ntvec)
    P0 = initial_density.reshape((drc, drc))
    propagated_dens = [P0]
    H0 = MLham.apply(hkparams,P0.real,P0.imag)
    H0 = H0[0] + 1j*H0[1]
    evals, evecs = jnp.linalg.eigh(H0)
    U0 = evecs @ jnp.diag(jnp.exp(-1j*dt*evals)) @ evecs.conj().T
    P1 = U0 @ P0 @ U0.conj().T
    propagated_dens.append( P1 )
    def bodyfun(i, dl):
        P0 = dl[i, :, :]
        P1 = dl[i+1, :, :]
        H1 = MLham.apply(hkparams,P1.real,P1.imag)
        H1 = H1[0] + 1j*H1[1]
        evals, evecs = jnp.linalg.eigh(H1)
        U1 = evecs @ jnp.diag(jnp.exp(-2j*dt*evals)) @ evecs.conj().T
        P2 = U1 @ P0 @ U1.conj().T
        return dl.at[i+2].set( P2 )
    
    alldens = jnp.concatenate([jnp.stack(propagated_dens), jnp.zeros((ntvec-1, drc, drc))], axis=0) 
    fdl = lax.fori_loop(0, ntvec-1, bodyfun, alldens)
    
    return fdl

def loss(hkparams, thisden, thistmeoff, thisfrq, thisamp, thisdirec):
    thisic = thisden[0,:,:].reshape((-1))
    mlprop = MMUT_Prop_HSB(hkparams, thisic, 
                           tmeoff=thistmeoff, fldfrq=thisfrq, fldamp=thisamp, norm_direc=thisdirec)
    resid = mlprop - thisden[:ntvec+1, :, :]
    mse = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))/ntvec
    return mse

# JIT AND VMAP AND ALL THAT JAZZ
jloss = jit(loss)
jadjgrad = jit(adjgrad)

jaggloss = soft_pmap(loss, in_axes=(None,0,0,0,0,0))
#jaggloss = jit(aggloss)

jaggadjgrad = soft_pmap(adjgrad, in_axes=(None,0,0,0,0,0))
#jaggadjgrad = jit(aggadjgrad)

# define the training set
trnind = np.arange(45,57,dtype=np.int16)
#trnind = np.arange(0,4,dtype=np.int16)
trnden = np.stack(allden)[trnind]
trntme = np.stack(alltme)[trnind]
trnfrq = np.stack(allfrq)[trnind]
trnamp = np.stack(allamp)[trnind]
trnnd = np.stack(norm_direc)[trnind]
jtrnden = jnp.array(trnden)
jtrntme = jnp.array(trntme)
jtrnfrq = jnp.array(trnfrq)
jtrnamp = jnp.array(trnamp)
jtrnnd = jnp.array(trnnd)

# THE PURPOSE OF THE FOLLOWING BLOCK IS TO DEFINE A FUNCTION
# THAT CAN BE USED TO EVALUATE THE OBJECTIVE FUNCTION 
# ON A RANDOM CLOUD OF THETAS; WE THEN PICK THE BEST SUCH THETA as theta0

#can omit this for code output, or set theta0 to be random numbers (now hkparams/params)
# def jaxobj(x):
#     return jnp.mean(aggloss(x,jtrnden,jtrntme,jtrnfrq,jtrnamp,jtrnnd))

# jaxobj = vmap(jaxobj,in_axes=0,out_axes=0)
# jjaxobj = jit(jaxobj)

# rng = np.random.default_rng(seed=42)
# numtheta = 10000
# theta0NP = 0.6*rng.standard_normal(size=2 * drc**4 * numtheta) - 0.3
# theta0JNP = jnp.array(theta0NP).reshape(numtheta, 2*drc**4)
# test = jjaxobj(theta0JNP)
# theta0 = np.array(theta0JNP[jnp.argmin(test),:])

i = 0
ic = allden[i][0,:,:].reshape((-1))
print('propagating trajectory: {}'.format(allfld[0]))
#thetatrue = jnp.concatenate([beta1true.reshape((-1)), gamma1true.reshape((-1))])
mlprop = MMUT_Prop_HSB(params, ic, tmeoff=alltme[i], fldfrq=allfrq[i], fldamp=allamp[i])
import pdb;pdb.set_trace()
plt.plot(jnp.real(mlprop[:,0,0]),color='red')
plt.plot(jnp.real(allden[i][:mynumsteps+1,0,0]),color='black')
plt.savefig('pretrain.pdf')
plt.close()

# WRAPPERS TO ENABLE USE OF SCIPY OPTIMIZERS
def siobj(x):
    hkparams = populator(params, np.array(x))
    return np.mean(jaggloss(hkparams,jtrnden,jtrntme,jtrnfrq,jtrnamp,jtrnnd))

def sigrad(x):
    hkparams = populator(params, np.array(x))
    thisgrad = jaggadjgrad(hkparams,jtrnden,jtrntme,jtrnfrq,jtrnamp,jtrnnd)
    return np.array(jnp.mean( thisgrad, axis=0 ))

# UNCOMMENT THE FOLLOWING BLOCK IF YOU WISH TO unit test the adjoint method
# jaxgradloss = grad(loss, 0)
# jaxres = jaxgradloss(theta0, allden[0], alltme[0], allfrq[0], allamp[0], norm_direc[0])
# myres = jadjgrad(theta0, allden[0], alltme[0], allfrq[0], allamp[0], norm_direc[0])
# print('|| adjgrad - jaxgrad ||:')
# print(jnp.linalg.norm(jaxres - myres))

# check loss (before training) against loss evaluated at true theta
#print('pretraining loss value: ' + str(siobj(theta0)))
#print('true theta loss value: ' + str(siobj(thetatrue)))

#print('all true theta losses: ')
# print(jaggloss(jnp.array(thetatrue),jnp.stack(allden),
#                             jnp.stack(alltme),jnp.stack(allfrq),
#                             jnp.stack(allamp),jnp.stack(norm_direc)))

# UNCOMMENT THE FOLLOWING BLOCK IF YOU WISH TO USE BFGS
# def mycb(x):
#     thisloss = siobj(x)
#     print("iter {:d} loss {:0.6e}".format(mycb.iteration, thisloss))
#     mycb.iteration += 1
# mycb.iteration = 0
# res = scipy.optimize.minimize( siobj, 
#                                x0 = theta0,
#                                method = 'bfgs',
#                                callback = mycb,
#                                jac = sigrad,
#                                options = {'disp': True, 'gtol': 1e-30} )
# trainedtheta = res.x
######################################################################

#UNCOMMENT THE FOLLOWING BLOCK IF YOU WISH TO USE L-BFGS-B
# res = scipy.optimize.minimize( siobj, 
#                                x0 = np.array(flattener(params)),
#                                method = 'L-BFGS-B',
#                                jac = sigrad,
#                                options = {'iprint': 1, 'ftol': 1e-30, 'gtol': 1e-30} )
# trainedtheta = res.x
# ######################################################################

# UNCOMMENT THE FOLLOWING BLOCK IF YOU WISH TO USE trust region with SR1 Hessian approximation
# res = scipy.optimize.minimize( siobj, 
#                                x0 = theta0,
#                                method = 'trust-constr',
#                                hess = scipy.optimize.SR1(),
#                                jac = sigrad,
#                                options = {'disp': True, 'verbose': 2, 'xtol': 1e-30, 'gtol': 1e-30} )
# trainedtheta = res.x
######################################################################

# UNCOMMENT THIS NEXT BLOCK IF YOU WANT TO USE OPTAX
def myobj(jx):
    hkparams = populator(params, jx)
    return jnp.mean(jaggloss(hkparams,jtrnden,jtrntme,jtrnfrq,jtrnamp,jtrnnd))

def mygrad(jx):
    hkparams = populator(params, jx)
    thisgrad = jaggadjgrad(hkparams,jtrnden,jtrntme,jtrnfrq,jtrnamp,jtrnnd)
    return jnp.mean( thisgrad, axis=0 )

def fit(params: optax.Params, optimizer: optax.GradientTransformation, nfs, dispint, saveint) -> optax.Params:
    opt_state = optimizer.init(params)

    def step(params, opt_state):
        grads = mygrad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for i in range(nfs):
        params, opt_state = step(params, opt_state)
        if i % dispint == 0:
            loss_value = myobj(params)
            print(f'step {i}, loss: {loss_value}')
        if i % saveint == 0:
            np.savez('optaxADJMMUTtheta.npz',trainedtheta=params)
            with open("optaxADJMMUTloss.txt",'a',encoding = 'utf-8') as f:
                f.write(f'step {i}, loss: {loss_value}\n')
    
    return params

optimizer = optax.fromage(learning_rate=1e-6)
opt_state = optimizer.init(flattener(params))
trainedtheta = fit(flattener(params), optimizer, 10000, 1, 10)
######################################################################

np.savez('trainedtheta.npz', trainedtheta=trainedtheta)

print("|| theta0 - truetheta ||")
print( jnp.linalg.norm(theta0-thetatrue) )

print("|| trainedtheta - truetheta ||")
print( jnp.linalg.norm(trainedtheta-thetatrue) )

print("jaggloss(truetheta) = " + str(siobj(thetatrue)))
print("jaggloss(trainedtheta) = " + str(siobj(trainedtheta)))

i = 10
ic = allden[i][0,:,:].reshape((-1))
mlprop = MMUT_Prop_HSB(trainedtheta, ic, tmeoff=alltme[i], fldfrq=allfrq[i], fldamp=allamp[i])
print(0.5*jnp.linalg.norm(mlprop - allden[i][:mynumsteps+1,:,:])**2)
plt.plot(jnp.real(mlprop[:,0,0]),color='red')
plt.plot(jnp.real(allden[i][:mynumsteps+1,0,0]),color='black')
plt.savefig('posttrain1.pdf')
plt.close()

plt.plot(jnp.real(mlprop[3900:4000,0,0]),color='red')
plt.plot(jnp.real(allden[i][3900:4000,0,0]),color='black')
plt.savefig('posttrain2.pdf')
plt.close()

def MMUT_Save_Ham(hkparams, initial_density, tmeoff=1, fldfrq=1, fldamp=1, norm_direc=jnp.array([0.0,0.0,1.0])):
    tvec = dt*jnp.arange(ntvec)
    P0 = initial_density.reshape((drc, drc))
    propagated_dens = [P0]
    H0 = MLham.apply(hkparams,P0.real,P0.imag)
    H0 = H0[0] + 1j*H0[1]
    propagated_hams = [H0]
    evals, evecs = jnp.linalg.eigh(H0)
    U0 = evecs @ jnp.diag(jnp.exp(-1j*dt*evals)) @ evecs.conj().T
    P1 = U0 @ P0 @ U0.conj().T
    propagated_dens.append( P1 )
    def bodyfun(i, intup):
        dl, hl = intup
        P0 = dl[i, :, :]
        P1 = dl[i+1, :, :]
        H1 = MLham.apply(hkparams,P1.real,P1.imag)
        H1 = H1[0] + 1j*H1[1]
        evals, evecs = jnp.linalg.eigh(H1)
        U1 = evecs @ jnp.diag(jnp.exp(-2j*dt*evals)) @ evecs.conj().T
        P2 = U1 @ P0 @ U1.conj().T
        return (dl.at[i+2].set( P2 ), hl.at[i+1].set( H1 ))
    
    alldens = jnp.concatenate([jnp.stack(propagated_dens), jnp.zeros((ntvec-1, drc, drc))], axis=0)
    allhams = jnp.concatenate([jnp.stack(propagated_hams), jnp.zeros((ntvec-1, drc, drc))], axis=0)
    fdh = lax.fori_loop(0, ntvec-1, bodyfun, (alldens, allhams))
    
    return fdh

i = 99
ic = allden[i][0,:,:].reshape((-1))
mynumsteps = 4000
mydenham = MMUT_Save_Ham(trainedtheta, ic, tmeoff=alltme[i], fldfrq=allfrq[i], fldamp=allamp[i])
trdenham = MMUT_Save_Ham(thetatrue, ic, tmeoff=alltme[i], fldfrq=allfrq[i], fldamp=allamp[i])

print("Hamiltonian errors")
print( jnp.mean(jnp.square(jnp.abs(mydenham[1][:,0,1] - trdenham[1][:,0,1]))) )
print( jnp.mean(jnp.square(jnp.abs(mydenham[1][:,1,0] - trdenham[1][:,1,0]))) )
print( jnp.mean(jnp.square(jnp.abs((mydenham[1][:,1,1] - mydenham[1][:,0,0]) - (trdenham[1][:,1,1] - trdenham[1][:,0,0])))) )

mycom = jnp.einsum('aij,ajk->aik',mydenham[1],mydenham[0][:-1,:,:]) - jnp.einsum('aij,ajk->aik',mydenham[0][:-1,:,:],mydenham[1])
trcom = jnp.einsum('aij,ajk->aik',trdenham[1],trdenham[0][:-1,:,:]) - jnp.einsum('aij,ajk->aik',trdenham[0][:-1,:,:],trdenham[1])
print("Commutator error: ")
print(jnp.mean(jnp.square(jnp.abs(mycom - trcom))))

