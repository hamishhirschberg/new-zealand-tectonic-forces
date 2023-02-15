#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hamish Hirschberg
"""

import numpy as np

def uniformStress(model, postProp=0., aveRange=None):
    """Adjusts strain rate capacity to produce uniform stress magnitudes.
    
    This function adjusts strain rate capacity to produce approximately
    uniform a priori stress magnitudes in all elements. Strain-rate
    capacities in elements with large apriori stress magnitudes are increased
    to make the elements weaker. Strain-rate capacities in elements with
    small a priori stress magnitudes are reduced to make the elements
    stronger.
    
    Parameters
    ----------
    model : permdefmap model
        Model where the strain-rate capacities are being adjusted. The
        a priori and a posteriori results need to have been read into
        the model but this function does not check for that.
    postProp : float or array of float, default=0.
        Perform adjustment based on a proportion of the a posteriori stress
        relative to a priori stress. A proportion of 0 uses just the
        a priori stress. A proportion of 1 uses the a posteriori stress.
        A proportion of 0.5 uses the average of the a priori and
        a posteriori stress. This allows the adjustment to be applied with
        another adjustment, e.g. to force potentials, with the proportion
        being the same for both adjustments.
    aveRange: range or array of int or array of bool, default=None
        Average B-value over this range of elements. Any variable that
        is accepted by a numpy array as an index can be used. Default
        averages over all elements.
    """
    if aveRange is None:
        aveRange = np.arange(model.nel)
    # A priori stress magnitude
    mag0 = np.sqrt(model.stressxx0**2 + model.stressyy0**2 
                  + 2*model.stressxy0**2)
    # Total area of all elements
    atot = np.sum(model.elarea[aveRange])
    # Sum of stress magnitudes, weighted by area
    magtot = np.sum(model.elarea[aveRange] * mag0[aveRange])
    
    # Calculate stress based on proportion of a posteriori used
    stressxx = (model.stressxx0[:model.nel]
                + model.stressxxd[:model.nel]*postProp[:model.nel])
    stressyy = (model.stressyy0[:model.nel]
                + model.stressyyd[:model.nel]*postProp[:model.nel])
    stressxy = (model.stressxy0[:model.nel]
                + model.stressxyd[:model.nel]*postProp[:model.nel])
    
    # New stress magnitude
    mag1 = np.sqrt(stressxx**2 + stressyy**2 + 2*stressxy**2)
    
    scale = np.divide(mag1*atot, magtot, where=magtot!=0.)
    # Scale strain-rate capacity
    model.straincapc[:model.nel] *= scale
    model.straincapcc[:model.nel] *= scale
    model.straincapcs[:model.nel] *= scale
    model.straincaps[:model.nel] *= scale
    model.straincapsc[:model.nel] *= scale
    model.straincapss[:model.nel] *= scale
    
def powerLaw(model, n, postProp=0., aveRange=None):
    """Adjusts strain rate capacity to produce power law rheology.
    
    This function adjusts strain rate capacity to approximately produce
    a power law rheology in all elements. The power law is
    T = B * E**(1/n)
    for stress magnitude T, strain rate magnitude E, power law exponent n,
    and scaling factor B.
    
    Parameters
    ----------
    model : permdefmap model
        Model where the strain-rate capacities are being adjusted. The
        a priori and a posteriori results need to have been read into
        the model but this function does not check for that.
    n : int
        Power law exponent.
    postProp : float or array of float, default=0.
        Perform adjustment based on a proportion of the a posteriori B-value
        relative to a priori B-value. A proportion of 0 uses just the
        a priori B-value. A proportion of 1 uses the a posteriori B-value.
        A proportion of 0.5 uses the average of the a priori and
        a posteriori B-value. This allows the adjustment to be applied with
        another adjustment, e.g. to force potentials, with the proportion
        being the same for both adjustments.
    aveRange: range or array of int or array of bool, default=None
        Average B-value over this range of elements. Any variable that
        is accepted by a numpy array as an index can be used. Default
        averages over all elements.
    """
    
    if aveRange is None:
        aveRange = np.arange(model.nel)
    # A priori stress magnitude
    tmag0 = np.sqrt(model.stressxx0**2 + model.stressyy0**2 
                  + 2*model.stressxy0**2)
    # A priori strain rate magnitude
    emag0 = np.sqrt(model.strainxx0**2 + model.strainyy0**2 
                  + 2*model.strainxy0**2)
    # A priori B value
    bmag0 = tmag0 * emag0**(-1/n)
    # Total area of all elements
    atot = np.sum(model.elarea[aveRange])
    # Sum of stress magnitudes, weighted by area
    magtot = np.sum(model.elarea[aveRange] * bmag0[aveRange])
    
    # Calculate stress based on proportion of a posteriori used
    stressxx = (model.stressxx0[:model.nel]
                + model.stressxxd[:model.nel]*postProp[:model.nel])
    stressyy = (model.stressyy0[:model.nel]
                + model.stressyyd[:model.nel]*postProp[:model.nel])
    stressxy = (model.stressxy0[:model.nel] 
                + model.stressxyd[:model.nel]*postProp[:model.nel])
    
    # Calculate strain rate based on proportion of a posteriori used
    strainxx = (model.strainxx0[:model.nel]
                + model.strainxxd[:model.nel]*postProp[:model.nel])
    strainyy = (model.strainyy0[:model.nel]
                + model.strainyyd[:model.nel]*postProp[:model.nel])
    strainxy = (model.strainxy0[:model.nel]
                + model.strainxyd[:model.nel]*postProp[:model.nel])
    
    # A posteriori stress magnitude
    tmag1 = np.sqrt(stressxx**2 + stressyy**2 + 2*stressxy**2)
    # A posteriori strain rate magnitude
    emag1 = np.sqrt(strainxx**2 + strainyy**2 + 2*strainxy**2)
    # A posteriori B value
    bmag1 = tmag1 * emag1**(-1/n)
    
    scale = np.divide(bmag1*atot, magtot, where=magtot!=0.)
    # Scale strain-rate capacity
    # Scale strain-rate capacity
    model.straincapc[:model.nel] *= scale
    model.straincapcc[:model.nel] *= scale
    model.straincapcs[:model.nel] *= scale
    model.straincaps[:model.nel] *= scale
    model.straincapsc[:model.nel] *= scale
    model.straincapss[:model.nel] *= scale
    
def byElement(model, prop=1., dstress=0., maxchange=1e6):
    """Adjusts strain-rate capacity on an element-by-element basis.
    
    This function adjusts the strain-rate capacity on an element-by-
    element basis based on the relative values of a posteriori and
    a priori strain rate. The adjustment is approximately the a posteriori
    strain rate divided by the a priori strain rate.
    
    Parameters
    ----------
    model : permdefmap model
        Model where the strain-rate capacities are being adjusted. The
        a priori and a posteriori results need to have been read into
        the model but this function does not check for that.
    prop : float, default=1.
        A value, typically between 0 and 1, indicating the proportion
        of the standard adjustment should be applied.
    dstress : float, default=0.
        The estimated change in stress from the adjustment. This allows
        the adjustment to more accurately calculated so that the new
        a priori solution will more closely match the old a posteriori
        solution.
    maxchange : float, default=1e6
        Maximum relative change. Useful for avoiding problems with small
        numbers.
    """
    # A priori for each element
    # Magnitude of stress * strain
    mag0 = (model.stressxx0*model.strainxx0 + model.stressyy0*model.strainyy0
            + 2*model.stressxy0*model.strainxy0)[:model.nel]
    # Correction if change in stress or force potentials
    if np.array(dstress).any():
        mag0 += (dstress[0,:model.nel]*model.strainxx0[:model.nel]
                 + dstress[1,:model.nel]*model.strainyy0[:model.nel]
                 + 2*dstress[2,:model.nel]*model.strainxy0[:model.nel])
    
    # A posteriori
    # Magnitude of a priori stress * a posteriori strain rate
    mag1 = (model.stressxx0*model.strainxx1 + model.stressyy0*model.strainyy1
            + 2*model.stressxy0*model.strainxy1)[:model.nel]
    # Prevent elements weakening when a posteriori and a priori signs differ
    mag1[mag1 < -mag0] = -mag0[mag1 < -mag0]
    # Calculate ratio of apost and apri strain rates for all elements
    ratio = np.abs(np.divide(mag1, mag0, where=mag0!=0.))
    ratio[ratio>maxchange] = maxchange
    ratio[ratio<1/maxchange] = 1/maxchange
    scale = ratio ** prop
    
    # Adjust strain-rate capacity on each element by its ratio
    model.straincapc[:model.nel] *= scale
    model.straincapcc[:model.nel] *= scale
    model.straincapcs[:model.nel] *= scale
    model.straincaps[:model.nel] *= scale
    model.straincapsc[:model.nel] *= scale
    model.straincapss[:model.nel] *= scale  