import numpy as np
import os
import re
import sys

class traj_data:
    #
    def __init__(self, data_file_path, molecule, basis, init_method, td_method, td_perturb, ind, npz_format=True,init_cond_file=None,td_dens_file=None,td_edip_file=None):
        #
        self.infilepath = data_file_path
        self.init_method = init_method
        self.mol = molecule
        self.bas = basis
        self.td_method = td_method
        self.td_perturb = td_perturb
        self.ind = ind
        #
        if (self.infilepath == ''):
            print('\nNo file-path specified! maybe try "./"?')
            print('Exiting...\n')
            return
        #
        if npz_format:
            icf = init_cond_file
            tdf = td_dens_file
            tef = td_edip_file
            aa = self.init_method
            cc = self.td_method
            bb = self.td_perturb
            prfx_lst = ['{}_s0'.format(aa), '{}_{}_s0'.format(aa,bb), '{}'.format(aa), '{}_{}'.format(aa,bb)]
            # read from initial condition data-file
            try:
                init_cond_files = [self.infilepath + '/ke+en+overlap+ee_twoe+dip_{}_{}_{}.npz'.format(ss,self.mol,self.bas) for ss in prfx_lst] 
                # print(init_cond_files)
                bool_lst = [os.path.isfile(f) for f in init_cond_files]
                index = bool_lst.index(True)
                init_datafile = init_cond_files[index]
                self.init_data = np.load(init_datafile, allow_pickle=True)
                # print(self.init_data.files)
            except Exception as e:
                print(e)
                init_datafile = self.infilepath + '/' + icf
                # print('\n\tLooking for "{}" instead...\n'.format(init_datafile))
                if os.path.isfile(init_datafile):
                    self.init_data = np.load(init_datafile, allow_pickle=True)
                    # print(self.init_data.files)
                else:
                    print('\n\tDesired time-dependent density matrix data-file not found. Exiting...\n')
                    return
            # read from time-dependent 1-density matrix data-files
            prfx_lst2 = ['{}_{}_s0'.format(cc,bb), '{}_{}'.format(cc,bb)]
            try:
                td_files = [self.infilepath + '/td_dens_re+im_{}_{}_{}.npz'.format(ss,self.mol,self.bas) for ss in prfx_lst2]
                # print(td_files)
                bool_lst2 = [os.path.isfile(f) for f in td_files]
                index2 = bool_lst2.index(True)
                td_dens_datafile = td_files[index2]
                # print('\n\treading from {}'.format(td_dens_datafile))
                self.td_dens_data = np.load(td_dens_datafile, allow_pickle=True)
            except Exception as e:
                print(e)
                td_dens_datafile = self.infilepath + '/' + tdf
                if os.path.isfile(td_dens_datafile):
                    # print('\n\treading from {}'.format(td_dens_datafile))
                    self.td_dens_data = np.load(td_dens_datafile, allow_pickle=True)
                else:
                    print('\n\tDesired time-dependent density matrix data-file not found. Exiting...\n')
                    return
            # read from time-dependent field and dipole moment data-files
            try:
                td_files = [self.infilepath + '/td_efield+dipole_{}_{}_{}.npz'.format(ss,self.mol,self.bas) for ss in prfx_lst2]
                # print(td_files)
                bool_lst2 = [os.path.isfile(f) for f in td_files]
                index2 = bool_lst2.index(True)
                td_fld_dip_datafile = td_files[index2]
                # print('\n\treading from {}\n'.format(td_fld_dip_datafile))
                self.td_fld_dip_data = np.load(td_fld_dip_datafile, allow_pickle=True)
            except Exception as e:
                print(e)
                td_fld_dip_datafile = self.infilepath + '/' + tef
                if os.path.isfile(td_fld_dip_datafile):
                    # print('\n\treading from {}\n'.format(td_fld_dip_datafile))
                    self.td_fld_dip_data = np.load(td_fld_dip_datafile, allow_pickle=True)
                else:
                    print('\n\tDesired time-dependent field and dipole moment data-file not found. Exiting...\n')
                    return
        else:
            print('\n\tOnly .NPZ format supported at the moment (please check the "npz_format" Boolean value). Exiting...\n')
        #
        return
    #
    def assign_init_data(self):
        #
        #print('returning kinmat, enmat, eeten, didat, xmat...')
        a = self.init_data
        #
        self.kinmat = a['ke_data']
        self.enmat = a['en_data']
        self.eeten = a['ee_twoe_data']
        self.drc = self.eeten.shape[0]
        #print('drc = {}'.format(self.drc))
        # exception handling because of how dipole moment matrix data is saved
        try:
            self.didat = np.array([a['dipx_data'],a['dipy_data'],a['dipz_data']])
        except:
            self.didat = np.array([a['dip_data'][0],a['dip_data'][1],a['dip_data'][2]])
        # needed for Loewdin orthogonalization
        self.s = a['overlap_data']
        self.sevals, self.sevecs = np.linalg.eigh(self.s)
        self.xmat = np.matmul(self.sevecs, np.diag(self.sevals**(-0.5)))
        #
        return
    #
    def assign_td_data(self, deltat, clean_step=100, cleanup=True):
        #
        cstep = clean_step
        self.dt = deltat
        cleanup = cleanup
        a = self.td_dens_data
        b = self.td_fld_dip_data
        # density matrix data
        denraw = a['td_dens_re_data'] + 1j*a['td_dens_im_data']
        denflat = denraw.reshape((-1,self.drc**2))
        # clean-up
        if cleanup == True:
            dennodupflat = np.array([np.delete(denflat[:,i], np.s_[(cstep+1)::cstep]) for i in range(self.drc**2)]).T
        else:
            dennodupflat = denflat
        denAO = dennodupflat.reshape((-1,self.drc,self.drc))
        # Loewdin orthogonalization
        self.denAO_ortho = np.diag(self.sevals**(0.5)) @ self.sevecs.T @ denAO @ self.sevecs @ np.diag(self.sevals**(0.5))
        # time-dependent field and dipole moment data
        self.td_fld = b['td_efield_data']
        self.td_dip = b['td_dipole_data']
        #
        self.timesteps = denAO.shape[0]
        #
        self.hfldAO_ortho = np.zeros((self.timesteps,self.drc,self.drc), np.complex128)
        #
        if self.td_perturb != 'delta':
            if self.didat.shape[0] == self.td_fld.shape[1]:
                for t in range(self.timesteps):
                    self.hfldAO_ortho[t] = np.sum(self.didat[:] * self.td_fld[t,:,None,None], axis=0)
                    self.hfldAO_ortho[t] = self.xmat.conj().T @ self.hfldAO_ortho[t] @ self.xmat
            elif self.didat.shape[0] == self.td_fld.shape[0]:
                for t in range(self.timesteps):
                    self.hfldAO_ortho[t] = np.sum(self.didat[:] * self.td_fld[:,t,None,None], axis=0)
                    self.hfldAO_ortho[t] = self.xmat.conj().T @ self.hfldAO_ortho[t] @ self.xmat
        #
        return
    #
    def clip_td_data(self, offset=2, tt=2000, ntrain=2000):
        #
        assert tt <= self.denAO_ortho.shape[0], '"tt" must be less than or equal to "self.denAO_ortho.shape[0]".'
        #
        self.offset = offset
        if tt < ntrain:
            print('''
            \t"tt" must be greater than or equal to "ntrain".
            \t Setting tt = ntrain...
            ''')
            self.tt = ntrain
        else:
            self.tt = tt
        self.ntrain = ntrain
        #
        den = self.denAO_ortho
        hfld = self.hfldAO_ortho
        #
        x_all = np.real(den)[self.offset:(self.tt+self.offset),:,:]
        y_all = np.imag(den)[self.offset:(self.tt+self.offset),:,:]
        hfld_all = hfld[self.offset:(self.tt+self.offset),:,:]
        #
        x_train = x_all[:self.ntrain,:,:]
        y_train = y_all[:self.ntrain,:,:]
        hfld_train = hfld_all[:self.ntrain,:,:] 
        # finite central difference for first derivative WRT time
        xdot = (-x_train[4:,:,:]*(1/12) + x_train[3:-1,:,:]*(2/3) - x_train[1:-3,:,:]*(2/3) + x_train[:-4,:,:]*(1/12))/self.dt
        ydot = (-y_train[4:,:,:]*(1/12) + y_train[3:-1,:,:]*(2/3) - y_train[1:-3,:,:]*(2/3) + y_train[:-4,:,:]*(1/12))/self.dt
        pdot = np.asarray(xdot + 1j*ydot)
        #
        xinp = np.asarray(x_train[2:-2,:,:])
        yinp = np.asarray(y_train[2:-2,:,:])
        #pinp = np.asarray(xinp + 1j*yinp)
        #pinpC = np.asarray(xinp - 1J*yinp)
        #pinpH = np.transpose(pinpC, axes=(0,2,1))
        hfldinp = np.asarray(hfld_train[2:-2,:,:])
        #
        return xinp, yinp, pdot, hfldinp, (x_all + 1j*y_all)
    
    def clip_our_td_data(self, ourpath, offset=2, tt=2000, ntrain=2000):
        prefix = 'td_dens_re+im_'
        bb = self.ind
        fname = ourpath+prefix+self.td_method+'_{}.npz'.format(bb)
        den = np.load(fname)['traj']
        self.offset = offset
        if tt < ntrain:
            print('''
            \t"tt" must be greater than or equal to "ntrain".
            \t Setting tt = ntrain...
            ''')
            self.tt = ntrain
        else:
            self.tt = tt
        self.ntrain = ntrain
        #
        hfld = self.hfldAO_ortho
        #
        x_all = np.real(den)[self.offset:(self.tt+self.offset),:,:]
        y_all = np.imag(den)[self.offset:(self.tt+self.offset),:,:]
        hfld_all = hfld[self.offset:(self.tt+self.offset),:,:]
        #
        x_train = x_all[:self.ntrain,:,:]
        y_train = y_all[:self.ntrain,:,:]
        hfld_train = hfld_all[:self.ntrain,:,:] 
        # finite central difference for first derivative WRT time
        xdot = (-x_train[4:,:,:]*(1/12) + x_train[3:-1,:,:]*(2/3) - x_train[1:-3,:,:]*(2/3) + x_train[:-4,:,:]*(1/12))/self.dt
        ydot = (-y_train[4:,:,:]*(1/12) + y_train[3:-1,:,:]*(2/3) - y_train[1:-3,:,:]*(2/3) + y_train[:-4,:,:]*(1/12))/self.dt
        pdot = np.asarray(xdot + 1j*ydot)
        #
        xinp = np.asarray(x_train[2:-2,:,:])
        yinp = np.asarray(y_train[2:-2,:,:])
        #pinp = np.asarray(xinp + 1j*yinp)
        #pinpC = np.asarray(xinp - 1J*yinp)
        #pinpH = np.transpose(pinpC, axes=(0,2,1))
        hfldinp = np.asarray(hfld_train[2:-2,:,:])
        #
        return xinp, yinp, pdot, hfldinp, (x_all + 1j*y_all)
    #

