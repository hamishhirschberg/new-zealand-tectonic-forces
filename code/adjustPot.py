#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hamish Hirschberg
"""

import numpy as np
    
def fromTractionStressWeighted(model, prop=1., change=False):
    """Adjusts force potentials using weighted traction and stress differences.
    
    This function adjusts force potentials with the aim of fitting
    observations exactly without consideration of the phyical likelihood
    of the resultant force potentials. The potentials at points on
    faults are adjusted based on the difference of a posteriori minus
    a priori tractions on adjacent fault segments, weighted by the slip
    rate capacity and length of each segment. Potentials at points off
    faults are adjusted based on the difference of a posteriori minus
    a priori stresses in adjacent elements, weighted by the strain rate
    capacity and area of the elements.
    
    Parameters
    ----------
    model : permdefmap model
        Model where the force potentials are being adjusted. The
        difference between a priori and a posteriori results need to
        have been read into the model but this function does not check
        for that.
    prop : float, default=1.
        A value, typically between 0 and 1, indicating the proportion
        of the standard adjustment that should be applied.
    change : bool, default=False
        If True, returns the change to force potentials.
    
    Returns
    -------
    change : array of float, optional
        Change to force potentials as an array of xx, yy, and xy components.
    """
    
    from ..geometry import strainCapMag, slipCapMag
    
    # Arrays of output force potential adjustments
    dpotxx = np.zeros((model.ngp))
    dpotyy = np.zeros((model.ngp))
    dpotxy = np.zeros((model.ngp))
    # Array of weight sum (area * cap)
    areacap = np.zeros((model.ngp))
    
    # Calculate force potential adjustments in elements
    dpotel = np.array([[model.stressxxd, model.stressxyd],
                       [model.stressxyd, model.stressyyd]])
#    # adjust for change in stress
#    if np.array(dstress).any():
#        dpotel[0,0,:]=2*dpotel[0,0,:]-dstress[0,:]
#        dpotel[1,1,:]=2*dpotel[1,1,:]-dstress[1,:]
#        dpotel[0,1,:]=2*dpotel[0,1,:]-dstress[2,:]
#        dpotel[1,0,:]=2*dpotel[1,0,:]-dstress[2,:]
    strainmag = strainCapMag(model)
    # Assign force potentials to gridpoints
    for e in range(model.nel):
        # Weighting contribution
        weight = model.elarea[e] * strainmag[e]
        areacap[model.gp1ofel[e]] += weight
        areacap[model.gp2ofel[e]] += weight
        areacap[model.gp3ofel[e]] += weight
        # Weighted force potentials
        dpotxx[model.gp1ofel[e]] += dpotel[0,0,e] * weight
        dpotxx[model.gp2ofel[e]] += dpotel[0,0,e] * weight
        dpotxx[model.gp3ofel[e]] += dpotel[0,0,e] * weight
        dpotyy[model.gp1ofel[e]] += dpotel[1,1,e] * weight
        dpotyy[model.gp2ofel[e]] += dpotel[1,1,e] * weight
        dpotyy[model.gp3ofel[e]] += dpotel[1,1,e] * weight
        dpotxy[model.gp1ofel[e]] += dpotel[0,1,e] * weight
        dpotxy[model.gp2ofel[e]] += dpotel[0,1,e] * weight
        dpotxy[model.gp3ofel[e]] += dpotel[0,1,e] * weight
    # Divide weighted sum by sum of weights to get average
    dpotxx /= areacap
    dpotyy /= areacap
    dpotxy /= areacap
    # Identify points on faults
    faults = model.nsegatgp[:model.ngp] > 0.5
    # Reset points on faults
    dpotxx[faults] = 0
    dpotyy[faults] = 0
    dpotxy[faults] = 0
    
    # Calculate force potential adjustments for faults
    dpottn = model.tractd
    dpotnn = model.tracnd
    dpottt = np.zeros_like(dpotnn)
    # Array of weight sum (length * cap)
    segcap = np.zeros((model.ngp))
#    if np.array(dtrac).any():
#        dpottn=2*dpottn-dtrac[0,:,:]
#        dpotnn=2*dpotnn-dtrac[1,:,:]
    
    slipmag = slipCapMag(model)
    for f in range(model.nfault):
        for s in range(model.nfaultseg[f]):
            # Rotation matrix for fault segment
            rot = np.array([[model.segtx[s,f], model.segty[s,f]],
                            [model.segnx[s,f], model.segny[s,f]]])
            # Calculate tt-component based on elements adjacent to segment
            dpote = (dpotel[:,:,model.el1onside[model.sideonfault[s,f]]]
                     + dpotel[:,:,model.el2onside[model.sideonfault[s,f]]]) * 0.5
            dpottt[s,f] = np.einsum('ij,i,j', dpote, rot[0,:], rot[0,:])
            # Rotate from tn to xy coordinates for segment
            dpotseg = np.einsum('ki,lj,kl', rot, rot,
                                [[dpottt[s,f], dpottn[s,f]],
                                 [dpottn[s,f], dpotnn[s,f]]])
            # Contribution to weight sum
            weight = model.seglength[s,f] * slipmag[s,f]
            segcap[model.gponfault[s,f]] += weight
            segcap[model.gponfault[s+1,f]] += weight
            # Assign force potentials to segments
            dpotxx[model.gponfault[s,f]] += dpotseg[0,0] * weight
            dpotxx[model.gponfault[s+1,f]] += dpotseg[0,0] * weight
            dpotyy[model.gponfault[s,f]] += dpotseg[1,1] * weight
            dpotyy[model.gponfault[s+1,f]] += dpotseg[1,1] * weight
            dpotxy[model.gponfault[s,f]] += dpotseg[1,0] * weight
            dpotxy[model.gponfault[s+1,f]] += dpotseg[1,0] * weight
    # Account for multiple segments at each point
    dpotxx[faults] /= segcap[faults]
    dpotyy[faults] /= segcap[faults]
    dpotxy[faults] /= segcap[faults]
    
    # Add to existing force potentials
    model.potxx[:model.ngp] += dpotxx * prop
    model.potyy[:model.ngp] += dpotyy * prop
    model.potxy[:model.ngp] += dpotxy * prop
    
    if change:
        return np.array([dpotxx, dpotyy, dpotxy]) * prop