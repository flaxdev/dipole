# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:49:12 2021

@author: flavio
"""
import numpy as np
from numpy import sin
from numpy import cos
from numpy import deg2rad
from scipy.optimize import dual_annealing, basinhopping, shgo, differential_evolution, root, fsolve, minimize, differential_evolution, NonlinearConstraint
from scipy.stats import f


class dipole:
    nu = 0.25
    
    coord = [0,0,-1000]
    azim = 0
    azenith = 0 # 0 vertical, 90 horizontal
    hypervolume = 1e9  # dV*d    
    kappa = 0
    
    confidence_coord = [[0, 0, -1000], [0, 0, 0]]
    confidence_azim = [0, 0]
    confidence_azenith = [0, 0]
    confidence_hypervolume = [-1e9, 1e9]
    confidence_kappa = [-1, 1]
    
    
    
    
    def __init__(self, coord, azim, azenith, hypervolume, k=0.0):
        self.coord = coord
        self.azim = azim
        self.azenith = azenith
        self.hypervolume = hypervolume
        self.kappa = k

    def __str__(self):
        return f'coords: {self.coord} \n azim: {self.azim} \n azenith: {self.azenith} \n hypervolume: {self.hypervolume} \n kappa: {self.kappa} \n'
    
    def __repr__(self):
        return f'coords: {self.coord} \n azim: {self.azim} \n azenith: {self.azenith} \n hypervolume: {self.hypervolume} \n kappa: {self.kappa} \n'


    def moment(self):
        # dipole moment

        m = (self.hypervolume*(1-self.nu)/np.pi)*np.array([sin(deg2rad(self.azim))*sin(deg2rad(self.azenith)), 
                                  cos(deg2rad(self.azim))*sin(deg2rad(self.azenith)), 
                                  cos(deg2rad(self.azenith))])        
        return m
    
    @staticmethod
    def displ(coord, azim, azenith, hypervolume, k, nu, points):
        
        # dipole moment
        m = (hypervolume*(1-nu)/np.pi)*np.array([sin(deg2rad(azim))*sin(deg2rad(azenith)), 
                                  cos(deg2rad(azim))*sin(deg2rad(azenith)), 
                                  cos(deg2rad(azenith))]) 
        
        r0 = np.array(points) - np.array(coord)  # referenced points

        # vertical positive downwards 
        m = -m
                       
        R = np.linalg.norm(r0,axis=1)
        r = np.array([rpi/Ri for rpi, Ri in zip(r0, R)])
        
        ez = np.array([0,0,-1])
        
        nablas = np.array([(np.identity(3) - 3*np.outer(ri,ri) + 
                            k* np.outer(r0i,ez)/r0i[2])/(Ri**3)  for r0i, ri, Ri in zip(r0,r,R)])
    
    
        u = np.array([np.dot(nabla, m) for nabla in nablas])
                        
        return u

    def u(self, points):
        
        return self.displ(self.coord, self.azim, self.azenith, self.hypervolume, self.kappa, self.nu, points)
    
    @staticmethod
    def tilt(coord, azim, azenith, hypervolume, nu, points):
        
        # dipole moment
        m = (hypervolume*(1-nu)/np.pi)*np.array([sin(deg2rad(azim))*sin(deg2rad(azenith)), 
                                  cos(deg2rad(azim))*sin(deg2rad(azenith)), 
                                  cos(deg2rad(azenith))]) 
        
        repoints = np.array(points) - np.array(coord)  # referenced points

        if repoints.ndim == 1:
            repoints = np.expand_dims(repoints, axis=0)
        # vertical positive downwards 
        m = -m
                       
        R = np.linalg.norm(repoints,axis=1)
        
        r = np.array([rpi/Ri for rpi, Ri in zip(repoints, R)])
        
        nablas = np.array([[[(-3/(Ri**4))*d*(4*x**2-y**2-d**2), (-3/(Ri**4))*5*x*y*d, (-3/(Ri**4))*x*(4*d**2-x**2-y**2) ], \
                          [(-3/(Ri**4))*d*x*y*d, (-3/(Ri**4))*d*(4*y**2-x**2-d**2), (-3/(Ri**4))*y*(4*d**2-x**2-y**2)]] for (x,y,d), Ri in zip(r, R)])   
    
        T = np.array([np.dot(nabla, m) for nabla in nablas])
                
        return T

    def t(self, points):
        
        return self.tilt(self.coord, self.azim, self.azenith, self.hypervolume, self.nu, points)

    # misfit only for displacements
    def misfit(self, points, displacements, displerrors, nu=0.25, err='wrmse'): 
        
        errfun = self.__errfunc(points, displacements, displerrors, nu, err)
        x0 = [self.coord[0], self.coord[1], self.coord[2], self.azim, self.azenith, self.hypervolume, self.kappa]
        e0 = errfun(x0)
        
        return e0
    
    # confidence interval calculated besed only on displacements
    def calculate_confidence_intervals(self, points, displacements, coord_ranges, azim_range, azenith_range, hypervolume_range, kappa_range, displerrors=1, nu=0.25, err='wrmse', alpha=0.05, maxiter=2000):
        
        
        errfunc = self.__errfunc(points=points, displacements=displacements, displerrors=displerrors, nu=nu, err=err)
        
        x0 = [self.coord[0], self.coord[1], self.coord[2], self.azim, self.azenith, self.hypervolume, self.kappa]
        e0 = errfunc(x0)
        
        m = 7 # model parameters
        n = np.size(displacements)
        
        Fk = f.ppf(q=1-alpha, dfn=m, dfd=n)
        ek = e0*(1+m*Fk/(n-m))
       
        nsteps = 10000
       
        ferrx = lambda a: (errfunc([a, x0[1], x0[2], x0[3], x0[4], x0[5], x0[6]])-ek)**2 
        vferrx = np.vectorize(ferrx)
        bnds = [(coord_ranges[0,0].tolist(), self.coord[0])]
        minx = differential_evolution(func=vferrx, x0=self.coord[0]-np.diff(bnds[0])[0]/nsteps , bounds=bnds)
        bnds = [( self.coord[0], coord_ranges[1,0].tolist())]
        maxx = differential_evolution(func=vferrx, x0=self.coord[0]+np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        self.confidence_coord[0][0] = minx.x[0]
        self.confidence_coord[1][0] = maxx.x[0]
        
        ferrx = lambda a: (errfunc([x0[0], a, x0[2], x0[3], x0[4], x0[5], x0[6]])-ek)**2 
        vferrx = np.vectorize(ferrx)
        bnds = [(coord_ranges[0,1].tolist(), self.coord[1])]
        minx = differential_evolution(func=vferrx, x0=self.coord[1]-np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        bnds = [( self.coord[1], coord_ranges[1,1].tolist())]
        maxx = differential_evolution(func=vferrx, x0=self.coord[1]+np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        self.confidence_coord[0][1] = minx.x[0]
        self.confidence_coord[1][1] = maxx.x[0]

        ferrx = lambda a: (errfunc([x0[0], x0[1], a, x0[3], x0[4], x0[5], x0[6]])-ek)**2 
        vferrx = np.vectorize(ferrx)
        bnds = [(coord_ranges[0,2].tolist(), self.coord[2])]
        minx = differential_evolution(func=vferrx, x0=self.coord[2]-np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        bnds = [( self.coord[2], coord_ranges[1,2].tolist())]
        print(bnds)
        maxx = differential_evolution(func=vferrx, x0=self.coord[2]+np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        self.confidence_coord[0][2] = minx.x[0]
        self.confidence_coord[1][2] = maxx.x[0]

        ferrx = lambda a: (errfunc([x0[0], x0[1], x0[2], a, x0[4], x0[5], x0[6]])-ek)**2 
        vferrx = np.vectorize(ferrx)
        bnds = [(azim_range[0], self.azim)]
        minx = differential_evolution(func=vferrx, x0=self.azim-np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        bnds = [( self.azim, azim_range[1])]
        maxx = differential_evolution(func=vferrx, x0=self.azim+np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        self.confidence_azim[0] = minx.x[0]
        self.confidence_azim[1] = maxx.x[0]

        ferrx = lambda a: (errfunc([x0[0], x0[1], x0[2], x0[3], a, x0[5], x0[6]])-ek)**2 
        vferrx = np.vectorize(ferrx)
        bnds = [(azenith_range[0], self.azenith)]
        minx = differential_evolution(func=vferrx, x0=self.azenith-np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        bnds = [( self.azenith, azenith_range[1])]
        maxx = differential_evolution(func=vferrx, x0=self.azenith+np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        self.confidence_azenith[0] = minx.x[0]
        self.confidence_azenith[1] = maxx.x[0]
        
        ferrx = lambda a: (errfunc([x0[0], x0[1], x0[2], x0[3], x0[4], a, x0[6]])-ek)**2 
        vferrx = np.vectorize(ferrx)
        bnds = [(hypervolume_range[0], self.hypervolume)]
        minx = differential_evolution(func=vferrx, x0=self.hypervolume-np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        bnds = [( self.hypervolume, hypervolume_range[1])]
        maxx = differential_evolution(func=vferrx, x0=self.hypervolume+np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        self.confidence_hypervolume[0] = minx.x[0]
        self.confidence_hypervolume[1] = maxx.x[0]

        ferrx = lambda a: (errfunc([x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], a])-ek)**2 
        vferrx = np.vectorize(ferrx)
        bnds = [(kappa_range[0], self.kappa)]
        minx = differential_evolution(func=vferrx, x0=self.kappa-np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        bnds = [( self.kappa, kappa_range[1])]
        maxx = differential_evolution(func=vferrx, x0=self.kappa+np.diff(bnds[0])[0]/nsteps, bounds=bnds)
        self.confidence_kappa[0] = minx.x[0]
        self.confidence_kappa[1] = maxx.x[0]

        return self
    
    @classmethod
    def __errfunc(cls, points, displacements, displerrors, nu, err): 
        
        errfunc = None
        
        if err=='wrmse':
            errfunc = lambda x: np.sqrt((((cls.displ(x[0:3], x[3], x[4], x[5], x[6], nu, points) - displacements)/displerrors) ** 2).mean())
            
        elif err=='relative':
            # relative error to avoid reference system
            def relarrange(values):
                matr = np.zeros((len(values), len(values), 3))
                for i in range(0, len(values)-1):
                    for j in range(i+1, len(values)):
                        for k in range(3):
                            matr[i,j,k] = values[i,k] - values[j,k]   
                return matr
            
            reldispl = relarrange(displacements)
            relerrdispl = np.ones((len(points), len(points), 3))
            if type(displerrors)!=list:
                displerrors = np.ones((len(points), 3))*displerrors
                
            for i in range(0, len(points)-1):
                for j in range(i+1, len(points)):
                    for k in range(3):
                        relerrdispl[i,j,k] = np.sqrt(displerrors[i,k]**2 + displerrors[j,k]**2)
                            
            errfunc = lambda x: np.sqrt((((relarrange(cls.displ(x[0:3], x[3], x[4], x[5], x[6], nu, points)) - reldispl)/relerrdispl) ** 2).mean())

        return errfunc
    
    
    # invert only for displacements
    @classmethod
    def invert(cls, points, displacements, coord_ranges, azim_range, azenith_range, hypervolume_range, k_range, displerrors,
                          coord0=None, azim0=None, azenith0=None, hypervolume0=None, k0=None, nu=0.25, alg='dual_annealing', err='wrmse'):
        
        errfunc = cls.__errfunc(points, displacements, displerrors, nu, err)              
        lw = [coord_ranges[0,0], coord_ranges[0,1], coord_ranges[0,2], azim_range[0], azenith_range[0], hypervolume_range[0], k_range[0]]        
        up = [coord_ranges[1,0], coord_ranges[1,1], coord_ranges[1,2], azim_range[1], azenith_range[1], hypervolume_range[1], k_range[1]]               
        
        bounds=np.array(list(zip(lw, up)))
        
        coord0 = coord0 if coord0 is not None else np.mean(bounds[:3,:], axis=1)
        azim0 = azim0 if azim0 is not None else np.mean(bounds[3,:])
        azenith0 = azenith0 if azenith0 is not None else np.mean(bounds[4,:])
        hypervolume0 = hypervolume0 if hypervolume0 is not None else np.mean(bounds[5,:])
        k0 = k0 if k0 is not None else np.mean(bounds[6,:])
                
        x0 = [coord0[0], coord0[1], coord0[2], azim0, azenith0, hypervolume0, k0]
        if alg=='dual_annealing':
            opt = dual_annealing(errfunc, bounds=bounds, x0 = x0)
        elif alg=='shgo':
            opt = shgo(errfunc, bounds=bounds, sampling_method='sobol', n=128)            
        elif alg=='differential_evolution':
            opt = differential_evolution(errfunc, bounds=bounds, maxiter=50000)
        elif alg=='basinhopping':
            minimizer_kwargs = {"method": "BFGS"}
            opt = basinhopping(errfunc, x0 = x0, niter=2000, minimizer_kwargs=minimizer_kwargs)
            
        elif alg=='all':            
            opt = shgo(errfunc, bounds=bounds, sampling_method='sobol', n=128)            
            x01 = opt.x
            opt = differential_evolution(errfunc, bounds=bounds, maxiter=50000)
            x02 = opt.x
            
            o1 = errfunc(x01)
            o2 = errfunc(x02)
            
            if o1<o2:
                x0 = x01
            else:
                x0 = x02

            opt = dual_annealing(errfunc, bounds=bounds, x0 = x0)
            x0 = opt.x

            minimizer_kwargs = {"method": "BFGS"}
            opt = basinhopping(errfunc, x0 = x0, niter=2000, minimizer_kwargs=minimizer_kwargs)
            
            
        
        return cls(opt.x[0:3], opt.x[3], opt.x[4], opt.x[5], opt.x[6])
        
    
class mogi:
    nu = 0.25
    
    coord = [0,0,-1000]
    volume = 1e9  # dV*d    
    
    confidence_coord = [[0, 0, -1000], [0, 0, -1000]]
    confidence_volume = [1e9, 1e9]
    
    
    
    def __init__(self, coord, volume):
        self.coord = coord
        self.volume = volume

    @staticmethod
    def displ(coord, volume, nu, points):
        
        
        repoints = np.array(points) - np.array(coord)  # referenced points
                       
        R = np.linalg.norm(repoints,axis=1)
                
        C = volume*(1-nu)/np.pi
        
        u = np.array([C*pi/(Ri**3) for pi, Ri in zip(repoints,R)])
                    
        return u

    def u(self, points):
        
        return self.displ(self.coord, self.volume, self.nu, points)

    @staticmethod
    def tilt(coord, volume, nu, points):
        
        
        repoints = np.array(points) - np.array(coord)  # referenced points

        if repoints.ndim == 1:
            repoints = np.expand_dims(repoints, axis=0)
                       
        R = np.linalg.norm(repoints,axis=1)
        
        T = 3*np.tile(repoints[:,2],(2,1)).T*repoints[:,:2]/np.tile(R**5,(2,1)).T
                        
        return T
    
    def t(self, points):
        
        return self.tilt(self.coord, self.volume, self.nu, points)

    @classmethod
    def __errfunc(cls, points, displacements, displerrors, nu, err): 
        
        errfunc = None
        
        if err=='wrmse':
            errfunc = lambda x: np.sqrt((((cls.displ(x[0:3], x[3], nu, points) - displacements)/displerrors) ** 2).mean())
            
        elif err=='relative':
            # relative error to avoid reference system
            def relarrange(values):
                matr = np.zeros((len(values), len(values), 3))
                for i in range(0, len(values)-1):
                    for j in range(i+1, len(values)):
                        for k in range(3):
                            matr[i,j,k] = values[i,k] - values[j,k]   
                return matr
            
            reldispl = relarrange(displacements)
            relerrdispl = np.ones((len(points), len(points), 3))
            if type(displerrors)!=list:
                displerrors = np.ones((len(points), 3))*displerrors
                
            for i in range(0, len(points)-1):
                for j in range(i+1, len(points)):
                    for k in range(3):
                        relerrdispl[i,j,k] = np.sqrt(displerrors[i,k]**2 + displerrors[j,k]**2)
                            
            errfunc = lambda x: np.sqrt((((relarrange(cls.displ(x[0:3], x[3], nu, points)) - reldispl)/relerrdispl) ** 2).mean())

        return errfunc


    # misfit only for displacements
    def misfit(self, points, displacements, displerrors, nu=0.25, err='wrmse'): 
        
        errfun = self.__errfunc(points, displacements, displerrors, nu, err)
        x0 = [self.coord[0], self.coord[1], self.coord[2], self.volume]
        e0 = errfun(x0)
        
        return e0

    # invert only for displacements
    @classmethod
    def invert(cls, points, displacements, coord_ranges, volume_range, displerrors,
                          coord0=[0,0,0], volume0=1e7, nu=0.25, alg='dual_annealing', err='wrmse'):
        
        errfunc = cls.__errfunc(points, displacements, displerrors, nu, err)              
        lw = [coord_ranges[0,0], coord_ranges[0,1], coord_ranges[0,2], volume_range[0]]        
        up = [coord_ranges[1,0], coord_ranges[1,1], coord_ranges[1,2], volume_range[1]]
        
        bounds=list(zip(lw, up))
                
        x0 = [coord0[0], coord0[1], coord0[2], volume0]
        if alg=='dual_annealing':
            opt = dual_annealing(errfunc, bounds=bounds, x0 = x0)
        elif alg=='shgo':
            opt = shgo(errfunc, bounds=bounds, sampling_method='sobol', n=128)            
        elif alg=='differential_evolution':
            opt = differential_evolution(errfunc, bounds=bounds, maxiter=50000)
        elif alg=='basinhopping':
            minimizer_kwargs = {"method": "BFGS"}
            opt = basinhopping(errfunc, x0 = x0, niter=2000, minimizer_kwargs=minimizer_kwargs)
            
        elif alg=='all':            
            opt = shgo(errfunc, bounds=bounds, sampling_method='sobol', n=128)            
            x01 = opt.x
            opt = differential_evolution(errfunc, bounds=bounds, maxiter=50000)
            x02 = opt.x
            
            o1 = errfunc(x01)
            o2 = errfunc(x02)
            
            if o1<o2:
                x0 = x01
            else:
                x0 = x02

            opt = dual_annealing(errfunc, bounds=bounds, x0 = x0)
            x0 = opt.x

            minimizer_kwargs = {"method": "BFGS"}
            opt = basinhopping(errfunc, x0 = x0, niter=2000, minimizer_kwargs=minimizer_kwargs)
            
            
        
        return cls(opt.x[0:3], opt.x[3])

