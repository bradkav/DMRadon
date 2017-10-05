import numpy as np
from scipy.special import erf

#----------------------------------------------------
# Calculate a normalisation factor for the distribution
def Nesc(sigv, vesc):
    return erf(vesc/(np.sqrt(2)*sigv)) \
        - np.sqrt(2.0/np.pi)*(vesc/sigv)*np.exp(-vesc**2/(2.0*sigv**2))


#----------------------------------------------------
# Calculate Maxwell-Boltzmann VELOCITY distribution 
# as a function of (v, theta, phi) where the polar angles
# are measured from the mean DM velocity, vlag
def VelDist(v, theta, phi, vlag=220.0, sigv=156.0, vesc=533.0):
    
    #Calculate overall normalisation
    prefactor = (2*np.pi*sigv*sigv)**-(1.5)
    normalisation = (prefactor/Nesc(sigv, vesc))
        
    #Calculate (v - vlag)^2
    dvsq = v**2 + vlag**2 - 2*v*vlag*np.cos(theta)
    vel_dist = normalisation*np.exp(-dvsq/(2*sigv**2))

    #Set to zero above the escape velocity, vesc
    if (np.isscalar(vel_dist)):
        if (dvsq > vesc**2):
            vel_dist = 0.0
    else:
        vel_dist[dvsq > vesc**2] = 0.0*vel_dist[dvsq > vesc**2]
    
    return vel_dist

#----------------------------------------------------
# Calculate Maxwell-Boltzmann SPEED distribution
# as a function of v
def SpeedDist(v, vlag=220.0, sigv=156.0, vesc=533.0):
    
    aplus = np.minimum((v+vlag), vesc)/(np.sqrt(2)*sigv)
    aminus = np.minimum((v-vlag), vesc)/(np.sqrt(2)*sigv)
    aesc = vesc/(np.sqrt(2)*sigv)
    
    #Calculate some normalisation factors
    normalisation = (v/(sigv*vlag*np.sqrt(2*np.pi)))
    normalisation /= Nesc(sigv, vesc)

    speed_dist = np.exp(-aminus**2) - np.exp(-aplus**2)
    
    if (np.isscalar(speed_dist)):
        if (v > vlag + vesc):
            speed_dist = 0.0
    else:
        speed_dist[v > vlag + vesc] = 0.0*speed_dist[v > vlag + vesc]
    return normalisation*speed_dist
        

#----------------------------------------------------
# Calculate the averaged velocity distribution
# i.e. average f(\vec{v}) over bins
# N_bins is the number of angular bins
# j = 1..N_bins is the index of the bin, 
# with j = 1 being the most forward
# Note that this is calculated by integrating over the angular
# bin then dividing by the angular size of the bin (i.e. average)
# so to obtain the speed distribution, you still need to 
# multiply by the angular size of each bin...
def VelDist_avg(v, j, N_bins,vlag=220.0, sigv=156.0, vesc=533.0):

    #Calculate some normalisation factors
    normalisation = ((v**(-1))/(sigv*vlag*np.sqrt(2*np.pi)))
    normalisation /= Nesc(sigv, vesc)

    #Calculate angular fraction/size of the bin
    frac = (np.cos((j-1)*np.pi*1.0/N_bins) - np.cos((j)*np.pi*1.0/N_bins))
    
    R = v*vlag/(sigv**2)
    d = ((v**2+vlag**2)/(2*sigv**2))
    A1 = np.maximum(np.cos(((j)-1)*np.pi*1.0/N_bins)+0.0*v, (v**2 + vlag**2 - vesc**2)/(2*v*vlag))
    A2 = np.maximum(np.cos((j)*np.pi*1.0/N_bins)+0.0*v,(v**2 + vlag**2 - vesc**2)/(2*v*vlag))
    
    vel_dist = np.exp(R*A1 - d) - np.exp(R*A2 - d)

    if (np.isscalar(vel_dist)):
        if ((v**2 + vlag**2 - vesc**2)/(2*v*vlag) > np.cos(((j)-1)*np.pi*1.0/N_bins)):
            vel_dist = 0.0
    else:
        vel_dist[(v**2 + vlag**2 - vesc**2)/(2*v*vlag) > np.cos(((j)-1)*np.pi*1.0/N_bins)] = 0.0*vel_dist[(v**2 + vlag**2 - vesc**2)/(2*v*vlag) > np.cos(((j)-1)*np.pi*1.0/N_bins)]
    
    return normalisation*vel_dist/(frac*2*np.pi)
    
    
#----------------------------------------------------
# Calculate the Maxwell-Boltzmann velocity integral
# eta = int_{vmin}^\infty f(\vec{v})/v d^3\vec{v}
# ^I hope you can parse latex in your head
def Eta(vmin, vlag=220.0, sigv=156.0, vesc=533.0):
    
    aplus = np.minimum((vmin+vlag), vesc)/(np.sqrt(2)*sigv)
    aminus = np.minimum((vmin-vlag), vesc)/(np.sqrt(2)*sigv)
    aesc = vesc/(np.sqrt(2)*sigv)
 
    vel_integral = (erf(aplus) - erf(aminus))
    vel_integral -= (2.0/(np.sqrt(np.pi)))*(aplus - aminus)*np.exp(-aesc**2)
    
    if (np.isscalar(vel_integral)):
        if (vmin > vesc + vlag):
            vel_integral = 0.0
    else:
        vel_integral[vmin > vesc + vlag] = 0.0*vel_integral[vmin > vesc + vlag] 
    
    return (1.0/(2.0*vlag*Nesc(sigv,vesc)))*vel_integral


#----------------------------------------------------
# Calculate the Radon Transform for the Maxwell-Boltzmann
# distribution in terms of vmin and (theta,phi), the polar angles
# from the median recoil direction (parallel to vlag).
# Note: there is no phi dependence, but we keep it in there for
# clarity.
def RadonTransform(vmin, theta, phi, vlag=220.0, sigv=156.0, vesc=533.0):
    
    #Calculate the squared-speed, as measured in the Galactic frame
    vhalosq = (vmin - vlag*np.cos(theta))**2
    #Some normalisation factors
    prefactor = 1.0/(Nesc(sigv, vesc)*sigv*np.sqrt(2.0*np.pi))
    RT = prefactor*(np.exp(-0.5*vhalosq/(sigv**2))-np.exp(-0.5*(vesc/sigv)**2.0))
    
    #Truncate at escape velocity
    if (np.isscalar(RT)):
        if (vhalosq > vesc**2):
            RT = 0.0
    else:
        RT[vhalosq > vesc**2] = RT[vhalosq > vesc**2]*0.0
    return RT

