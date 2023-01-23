import numpy as np

outpath = './nonlinhamtest/'

tprop = 19802
dt = 0.08268/10
drc = 2

# assume p is of size drc x drc    #
def EXham(t,p,fldfrq,fldamp,tmeoff,norm_direc,field=False):
    h = 0.5*(p**2)
    return h

def MMUT_Prop(hamfunc, initial_density, dt=0.08268, ntvec=2000, dilfac=1,
    field=False, tmeoff=1, fldfrq=1, fldamp=1, norm_direc=np.array([0.0,0.0,1.0])):
    tvec = dt*np.arange(ntvec)
    # set the dt of the code equal to dt of the data / dilation factor
    dtuse = dt/dilfac
    propagated_dens = [initial_density]
    for i in range(dilfac*ntvec-2):
        if i == 0:
            P0 = initial_density.reshape((drc, drc))                               # keep as a matrix
            H0 = hamfunc(i*dtuse,P0,fldfrq,fldamp,tmeoff,norm_direc,field)     # returns a matrix
            lamb, vmat = np.linalg.eigh(H0)
            U0 = vmat @ np.diag(np.exp(-1j*dtuse*lamb)) @ vmat.conj().T
            P1 = U0 @ P0 @ U0.conj().T
            propagated_dens.append( P1.reshape((drc**2)) )
        else:
            P0 = P1
            P1 = P2
        #
        H1 = hamfunc((i+1.0)*dtuse,P1,fldfrq,fldamp,tmeoff,norm_direc,field) # still a matrix!
        lamb, vmat = np.linalg.eigh(H1)
        U1 = vmat @ np.diag(np.exp(-2j*dtuse*lamb)) @ vmat.conj().T
        P2 = U1 @ P0 @ U1.conj().T
        if (i+1) % dilfac == 0:
            ii = (i+1)//dilfac
            propagated_dens.append( P2.reshape((drc**2)) )
    return np.stack(propagated_dens).reshape((-1,drc,drc))

# propagation
mynumsteps = 20001
print(f'Propagating for {mynumsteps} steps...',flush=True)
trajs = []
for i in range(100):     
    # training data
    N = 2
    # ic = np.random.normal(size=(N,N)) + 1.0j*np.random.normal(size=(N,N))
    # ic_herm = 0.5*(ic + ic.conj().T)
    # ic_herm = ic_herm.reshape((-1))
    diag = np.diag(np.random.dirichlet(np.ones(2),size=1).reshape(2,))
    offdiag1, offdiag2 = np.random.normal(size=1)[0], np.random.normal(size=1)[0]
    diag[0,1] = offdiag1
    diag[1,0] = offdiag2
    #N = 2
    #ic = np.random.normal(size=(N,N))
    ic_herm = (diag + diag.T)/2
    ic_herm = ic_herm.flatten()
    #import pdb; pdb.set_trace()
    print('')
    print('propagating trajectory')
    exprop = MMUT_Prop(EXham, ic_herm, dt=dt, ntvec=mynumsteps)
    fname=outpath+'traj_'+str(i)+'.npz'
    print(fname)
    np.savez(fname, traj=exprop.reshape((-1,drc,drc)))
    trajs.append(exprop.reshape((-1,drc,drc)))
    print('')
with open('nonlinearham.npz','wb') as f:
    np.save(f, np.array(trajs))