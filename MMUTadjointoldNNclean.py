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
#import haiku as hk


dt = 0.82680/10
#dt = 0.82680
mynumsteps = 2000
#mynumsteps = 19999
ntvec = mynumsteps

drc = 2


m = drc
print('reading new data')
allden = np.load('heh+_training_data_unitary.npz')
print('done reading new data')

# assume p is of size drc x drc    #
#latter parameters are for forcing, aka everything past p
nlayers = 4
layerwidths = [2*drc**2,256,256,256,2*drc**2]
numparams = 0
numweights = 0
for j in range(nlayers):
    numparams += layerwidths[j]*layerwidths[j+1] + layerwidths[j+1]
    numweights += layerwidths[j]*layerwidths[j+1]
def MLhamNN(theta,x,y): #t, fldfrq,fldamp,tmeoff,norm_direc):
    filt = []
    si = 0
    ei = layerwidths[0]*layerwidths[1]
    filt.append( theta[si:ei].reshape((layerwidths[0],layerwidths[1])) )
    si += layerwidths[0]*layerwidths[1]
    ei += layerwidths[1]*layerwidths[2]
    filt.append( theta[si:ei].reshape((layerwidths[1],layerwidths[2])) )
    si += layerwidths[1]*layerwidths[2]
    ei += layerwidths[2]*layerwidths[3]
    filt.append( theta[si:ei].reshape((layerwidths[2],layerwidths[3])) )
    si += layerwidths[2]*layerwidths[3]
    ei += layerwidths[3]*layerwidths[4]
    filt.append( theta[si:ei].reshape((layerwidths[3],layerwidths[4])) )
    bias = []
    si += layerwidths[3]*layerwidths[4]
    ei += layerwidths[1]
    bias.append( theta[si:ei] )
    si += layerwidths[1]
    ei += layerwidths[2]
    bias.append( theta[si:ei] )
    si += layerwidths[2]
    ei += layerwidths[3]
    bias.append( theta[si:ei] )
    si += layerwidths[3]
    ei += layerwidths[4]
    bias.append( theta[si:ei] )
    inplyr = jnp.array( jnp.concatenate([x.flatten(), y.flatten()]))
    #import pdb;pdb.set_trace()
    h1 = jax.nn.tanh( inplyr @ filt[0] + bias[0] )
    h2 = jax.nn.tanh( h1 @ filt[1] + bias[1] )
    h3 = jax.nn.tanh( h2 @ filt[2] + bias[2] )
    h4 = h3 @ filt[3] + bias[3]
    hreal = h4[0:drc**2]
    himag = h4[drc**2:]
    symmetrized_real = 0.5*(hreal.reshape((drc,drc)) + hreal.reshape((drc,drc)).T) 
    antisymmetrized_imag = 0.5*(himag.reshape((drc,drc)) - himag.reshape((drc,drc)).T)
    return symmetrized_real, antisymmetrized_imag


mydHdtheta = jacobian(MLhamNN, 0)
mydHdX = jacobian(MLhamNN, 1)
mydHdY = jacobian(MLhamNN,2)


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
    prederivamat = jnp.einsum('ij,jka,kl->ila',u.conj().T,w,u) 
    derivamat = jnp.einsum('ila,il->ila',prederivamat,mask)
    return jnp.einsum('ij,jka,kl->ila',u,derivamat,u.conj().T)

def xicomp(theta, x, y, evals, evecs):
    jacR = mydHdX(theta, x, y)
    jacI =  mydHdY(theta, x, y)
    dHdp = 0.5 * ((jacR[0] + 1.0j*jacR[1]) - 1.0j*(jacI[0] +1.0j*jacI[1]))
    dHdPbar = 0.5 * ((jacR[0] + 1.0j*jacR[1]) + 1.0j*(jacI[0] +1.0j*jacI[1]))
    dHdp = dHdp.reshape((drc,drc,drc,drc))
    dHdPbar = dHdPbar.reshape((drc,drc,drc,drc))
    jacP = expderiv2(evals, evecs, dHdp)
    jacPbar = expderiv2(evals, evecs, dHdPbar)
    return jacP, jacPbar

def dUdtheta(theta, x, y, evals, evecs):
    dHdtheta =  mydHdtheta(theta, x, y)
    tm1 = expderiv3(evals, evecs, dHdtheta[0])
    tm2 = expderiv3(evals, evecs, dHdtheta[1])
    return tm1 + 1j*tm2

def adjgrad(theta, Ptilde):
    tvec = dt*jnp.arange(ntvec)
    P0 = Ptilde[0,:,:]
    propagated_dens = [P0]
    H0 = MLhamNN(theta, P0.real, P0.imag)
    H0 = H0[0] + 1.0j*H0[1]
    evals, evecs = jnp.linalg.eigh(H0)
    U0 = evecs @ jnp.diag(jnp.exp(-1.0j*dt*evals)) @ evecs.conj().T
    P1 = U0 @ P0 @ U0.conj().T
    propagated_dens.append( P1 )
    # 
    def bodyfun(i, dtup):
        dl, dvals, dvecs, dU = dtup
        P0 = dl[i, :, :]
        P1 = dl[i+1, :, :]
        H1 = MLhamNN(theta,P1.real,P1.imag)
        H1 = H1[0] + 1.0j*H1[1]
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
        xiak = xicomp(theta, Pstack[k].reshape((-1)).real, Pstack[k].reshape((-1)).imag,-2j*dt*allevals[k], allevecs[k])
        xibk = xicomp(theta, Pstack[k].reshape((-1)).real, Pstack[k].reshape((-1)).imag, 2j*dt*allevals[k], allevecs[k])
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
    xiak = xicomp(theta, Pstack[ntvec-1].reshape((-1)).real, Pstack[ntvec-1].reshape((-1)).imag, -2j*dt*allevals[ntvec-1], allevecs[ntvec-1])
    xibk = xicomp(theta, Pstack[ntvec-1].reshape((-1)).real, Pstack[ntvec-1].reshape((-1)).imag,  2j*dt*allevals[ntvec-1], allevecs[ntvec-1])
    lambnext += jnp.einsum('ij,ijkl->kl',lambfinal @ allU[ntvec-1] @ Pstack[ntvec-2], (-2j*dt*xiak[0]).conj())
    lambnext += jnp.einsum('ij,ijkl->kl',lambfinal @ allU[ntvec-1] @ Pstack[ntvec-2], (-2j*dt*xiak[1]).conj()).conj()
    lambnext += jnp.einsum('ij,ijkl->kl',Pstack[ntvec-2] @ allU[ntvec-1].conj().T @ lambfinal, (2j*dt*xibk[0]).conj())
    lambnext += jnp.einsum('ij,ijkl->kl',Pstack[ntvec-2] @ allU[ntvec-1].conj().T @ lambfinal, (2j*dt*xibk[1]).conj()).conj()
    lambcat = jnp.concatenate([jnp.expand_dims(lambfinal,0), jnp.expand_dims(lambnext, 0), jnp.zeros((ntvec-2, drc, drc))])
    alllamb = lax.fori_loop(2, ntvec, bodylamb, lambcat)
    lambstack = jnp.flipud( jnp.stack(alllamb, axis=0) )    
    # 
    def bodygrad(k, gL):
        tmp1 = -2j*dt*(dUdtheta(theta, Pstack[k].real, Pstack[k].imag,-2j*dt*allevals[k], allevecs[k]))
        term1 = jnp.einsum('ija,jk,kl->ila',tmp1,Pstack[k-1],allU[k].conj().T)
        term2 = term1.transpose((1,0,2)).conj()
        return gL + jnp.real(jnp.einsum('il,ila->a',lambstack[k],(term1+term2).conj()))
    # 
    tmp1 = -1j*dt*(dUdtheta(theta, Pstack[0].real,  Pstack[0].imag, -1j*dt*allevals[0], allevecs[0]))
    term1 = jnp.einsum('ija,jk,kl->ila',tmp1,Pstack[0],allU[0].conj().T)
    term2 = term1.transpose((1,0,2)).conj()
    initgradL = jnp.real(jnp.einsum('il,ila->a',lambstack[0],(term1+term2).conj()))
    gradL = lax.fori_loop(1, ntvec, bodygrad, initgradL)/ntvec
    return gradL
    
def MMUT_Prop_HSB(theta, initial_density):
    tvec = dt*jnp.arange(ntvec)
    P0 = initial_density.reshape((drc, drc))
    propagated_dens = [P0]
    H0 = MLhamNN(theta, P0.real, P0.imag)
    H0 = H0[0] + 1.0j*H0[1]
    evals, evecs = jnp.linalg.eigh(H0)
    U0 = evecs @ jnp.diag(jnp.exp(-1.0j*dt*evals)) @ evecs.conj().T
    P1 = U0 @ P0 @ U0.conj().T
    propagated_dens.append( P1 )
    def bodyfun(i, dl):
        P0 = dl[i, :, :]
        P1 = dl[i+1, :, :]
        H1 = MLhamNN(theta, P1.real, P1.imag)
        H1 = H1[0] + 1.0j*H1[1]
        evals, evecs = jnp.linalg.eigh(H1)
        U1 = evecs @ jnp.diag(jnp.exp(-2j*dt*evals)) @ evecs.conj().T
        P2 = U1 @ P0 @ U1.conj().T
        return dl.at[i+2].set( P2 )
    
    alldens = jnp.concatenate([jnp.stack(propagated_dens), jnp.zeros((ntvec-1, drc, drc))], axis=0) 
    fdl = lax.fori_loop(0, ntvec-1, bodyfun, alldens)
    
    return fdl

def loss(theta, thisden):
    thisic = thisden[0,:,:].reshape((-1))
    mlprop = MMUT_Prop_HSB(theta, thisic, )
    resid = mlprop - thisden[:ntvec+1, :, :]
    mse = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))/ntvec
    return mse

# JIT AND VMAP AND ALL THAT JAZZ
jloss = jit(loss)
jadjgrad = jit(adjgrad)

aggloss = vmap(loss, in_axes=(None,0), out_axes=0)
jaggloss = jit(aggloss)

aggadjgrad = vmap(adjgrad, in_axes=(None,0), out_axes=0)
jaggadjgrad = jit(aggadjgrad)

#jaggloss = soft_pmap(loss, in_axes=(None,0,0,0,0,0))
#jaggloss = jit(aggloss)

#jaggadjgrad = soft_pmap(adjgrad, in_axes=(None,0,0,0,0,0))
#jaggadjgrad = jit(aggadjgrad)

# define the training set
trnind = np.arange(0,1,dtype=np.int16)
#trnind = np.arange(0,12,dtype=np.int16)
trnden = np.stack(allden)[trnind]
jtrnden = jnp.array(trnden)

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
print('propagating trajectory:')# {}'.format(allfld[0]))
#thetatrue = jnp.concatenate([beta1true.reshape((-1)), gamma1true.reshape((-1))])
rng = np.random.default_rng(seed=42)
theta0 = 0.6*rng.standard_normal(size=numparams) - 0.3
# filtinit = []
# si = 0
# ei = layerwidths[0]*layerwidths[1]
# sd0 = np.sqrt(2.0 / (layerwidths[0] + layerwidths[1]))
# filtinit.append( np.random.normal(loc=0,scale=sd0,size=ei-si) )
# si += layerwidths[0]*layerwidths[1]
# ei += layerwidths[1]*layerwidths[2]
# sd1 = np.sqrt(2.0 / (layerwidths[1] + layerwidths[2]))
# filtinit.append( np.random.normal(loc=0,scale=sd1,size=ei-si) )
# si += layerwidths[1]*layerwidths[2]
# ei += layerwidths[2]*layerwidths[3]
# sd2 = np.sqrt(2.0 / (layerwidths[2] + layerwidths[3]))
# filtinit.append( np.random.normal(loc=0,scale=sd2,size=ei-si) )
# si += layerwidths[2]*layerwidths[3]
# ei += layerwidths[3]*layerwidths[4]
# sd3 = np.sqrt(2.0 / (layerwidths[3] + layerwidths[4]))
# filtinit.append( np.random.normal(loc=0,scale=sd3,size=ei-si) )
# filtinit = jnp.concatenate(filtinit)
# # append zeros for biases
# theta0 = jnp.concatenate([filtinit, np.zeros(numparams - filtinit.shape[0])])
# print(theta0.shape[0] - numparams)
# print(theta0.size)
# print(numparams)
# np.save('mmutadj_heh+_unitary_theta0.npy',theta0)
mlprop = MMUT_Prop_HSB(theta0, ic)
plt.plot(jnp.real(mlprop[:,0,0]),color='red')
plt.plot(jnp.real(allden[i][:mynumsteps+1,0,0]),color='black')
plt.savefig('pretrain.pdf')
plt.close()

# WRAPPERS TO ENABLE USE OF SCIPY OPTIMIZERS
def siobj(x):
    #hkparams = populator(params, np.array(x))
    return np.mean(jaggloss(x,jtrnden))

def sigrad(x):
    #hkparams = populator(params, np.array(x))
    thisgrad = jaggadjgrad(x,jtrnden)
    return np.array(jnp.mean( thisgrad, axis=0 ))

# UNCOMMENT THE FOLLOWING BLOCK IF YOU WISH TO unit test the adjoint method
jaxgradloss = grad(loss, 0)
jaxres = jaxgradloss(theta0, allden[0])
myres = jadjgrad(theta0, allden[0])
#import pdb;pdb.set_trace()
print('|| adjgrad - jaxgrad ||:')
print(jnp.linalg.norm(jaxres - myres))

# #check loss (before training) against loss evaluated at true theta
# print('pretraining loss value: ' + str(siobj(theta0)))
# print('true theta loss value: ' + str(siobj(thetatrue)))

# print('all true theta losses: ')
# print(jaggloss(jnp.array(thetatrue),jnp.stack(allden),
#                             jnp.stack(alltme),jnp.stack(allfrq),
#                             jnp.stack(allamp),jnp.stack(norm_direc)))

# #UNCOMMENT THE FOLLOWING BLOCK IF YOU WISH TO USE BFGS
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
res = scipy.optimize.minimize( siobj, 
                               x0 = np.array(theta0),
                               method = 'L-BFGS-B',
                               jac = sigrad,
                               options = {'iprint': 1, 'ftol': 1e-30, 'gtol': 1e-30} )
trainedtheta = res.x
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
# def myobj(jx):
#     return jnp.mean(jaggloss(jx,jtrnden))

# def mygrad(jx):
#     thisgrad = jaggadjgrad(jx,jtrnden)
#     return jnp.mean( thisgrad, axis=0 )

# def fit(params: optax.Params, optimizer: optax.GradientTransformation, nfs, dispint, saveint) -> optax.Params:
#     opt_state = optimizer.init(params)

#     def step(params, opt_state):
#         grads = mygrad(params)
#         updates, opt_state = optimizer.update(grads, opt_state, params)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state

#     for i in range(nfs):
#         params, opt_state = step(params, opt_state)
#         if i % dispint == 0:
#             loss_value = myobj(params)
#             print(f'step {i}, loss: {loss_value}')
#         if i % saveint == 0:
#             np.savez('optaxmmutadj_rotated.npz',trainedtheta=params)
#             with open("optaxmmutadj_rotated.txt",'a',encoding = 'utf-8') as f:
#                 f.write(f'step {i}, loss: {loss_value}\n')
    
#     return params

# optimizer = optax.fromage(learning_rate=1e-4)
# opt_state = optimizer.init(jnp.array(theta0))
# trainedtheta = fit(jnp.array(theta0), optimizer, 10000, 1, 10)
######################################################################

np.savez('trainedtheta.npz', trainedtheta=trainedtheta)

#print("|| theta0 - truetheta ||")
#print( jnp.linalg.norm(theta0-thetatrue) )

#print("|| trainedtheta - truetheta ||")
#print( jnp.linalg.norm(trainedtheta-thetatrue) )

#print("jaggloss(truetheta) = " + str(siobj(thetatrue)))
#print("jaggloss(trainedtheta) = " + str(siobj(trainedtheta)))

i = 10
ic = allden[i][0,:,:].reshape((-1))
mlprop = MMUT_Prop_HSB(trainedtheta, ic)
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

