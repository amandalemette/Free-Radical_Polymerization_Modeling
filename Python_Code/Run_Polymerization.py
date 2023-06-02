import numpy as np
import os
import sys
from scipy.integrate import ode
from graphs_plot import plotgraphs

# ================================================================================== #
# ================================ INITIAL PARAMETERS ============================== #
# ================================================================================== #

global Rcte, V0 ,Vantes, Mji, Mjm, Mjp, Xc0, jc0, delta, Na, Vm_esp, Vp_esp, Vi_esp,T

# Integration interval definition
t0 = 0.0        # [min] reaction initial time
tf = 100        # [min] reaction end time
Nt = 8*tf       # [min] quantity of integration interval
tArray = np.linspace(t0, tf, Nt) # Time vector


# Operational conditions
T = 70          # [oC] Temperature
T = T + 273.15  # [K] Temperature
V0 = 1          # [L] Solution volume
# Monomer properties
Mjm = 100.13    # [g/mol] monomer molecular weight
rhom = 0.968 - 1.225*(1.e-3)*(T-273.15) # [g/cm3] monomer density (T in oC)
rhom = rhom*1000                       # [g/L] monomer density
Vm_esp  = 0.822 # [cm3/g]

# Polymer properties
Mjp = 150       # [g/mol]
Vp_esp  = 0.77  # [cm3/g]

# Initiator properties (AIBN)
Mji = 68            # [g/mol] initiator molecular weight
Vi_esp = 0.913  # [cm3/g]

# Other parameters
Rcte = 1.982        # Gas constant [cal/(mol.K)]
Xc0 = 100           # [ADM]
jc0 = 0.874         # [ADM]
delta = 6.9*(1.e-8) # [cm]
Na = 6.032*(1.e23)  # [1/mol]

# OED initial values

I0 = 0.01548    # [mol/L] initial initiator concentration
Vm0 = V0        # [L] initial monomer volume (the chosen value doesnt affect the program as all parameters are in terms of concentration and the initial solution volume is the initial monomer volume)
M0 = rhom/Mjm   # [mol/L] initial monomer concentration

NInputVar = 6   # number of input variables
InputVar = np.zeros(NInputVar)     # input variables vector for integration

InputVar[0] = V0    # [L] V           (solution volume)
InputVar[1] = M0    # [mol/L] M       (monomer concentration)
InputVar[2] = I0    # [mol/L] I       (initiator concentration)
InputVar[3] = 1.e-20   # [mol/L] mu0     (dead polymer chain)
InputVar[4] = 1.e-20   # [mol/L] mu1
InputVar[5] = 1.e-20   # [mol/L] mu2



# ================================================================================== #
# ============================= KINETIC CONST FUNCTIONS ============================ #
# ================================================================================== #

def fkp(T,lambda0,wm,phip):             # T in [K]
    global Rcte, Mjm, Mjp, Xc0, jc0, delta, Na, Vm_esp, Vp_esp
    
    kp0 = 4.92*(1.e5)*np.exp(-4353/(Rcte*T))*60 # [L/(mol.min)]

    invjc = 1/jc0 + 2*phip/Xc0  # [ADM]
    jc = 1/invjc                # [ADM]
    tau = np.sqrt(3/(2*jc*(delta ** 2)))    # [1/cm]
    rt = np.sqrt(np.log(1000*(tau ** 3)/(Na*lambda0*(3.1415 ** (1.5)))))/tau # [cm] (1000 converts L -> cm3)
    
    rm = rt

    wp = 1-wm       # [ADM]
    gamma = 0.763   # [ADM]
    Vfm = 0.149 + 2.9*(1.e-4)*(T-273.15)
    Vfp = 0.0194 + 1.3*(1.e-4)*(T-273.15 - 105)
    Vf = wm*Vm_esp*Vfm + wp*Vp_esp*Vfp
    factor = -gamma*Vm_esp*Mjm*(wm/Mjm + wp/Mjp)/Vf #[ADM]
    Dm0 = 0.827*(1.e-10)    # [cm2/s]
    Dm = Dm0*np.exp(factor) # [cm2/s]
    Dm = Dm*60              # [cm2/min]

    tauDp = (rm ** 2)/(3*Dm)    # [min]
    tauRp = 1/(kp0*lambda0)     # [min]

    kp = kp0/(1+tauDp/tauRp)

    return kp          # [L/(mol.min)]

def fktc(T,lambda0,phip,Mw,wm,kp,M):            # T in [K]
    global Rcte, Mji, Mjp, Xc0, jc0, delta, Na, Vm_esp, Vp_esp, Vi_esp
    
    ktc0 = 9.80*(1.e7)*np.exp(-701.0/(Rcte*T))*60   # [L/(mol.min)]

    Mwlin = 4293.76         # [g/mol]
    kB = 1.3806*(1.e-23)    # [J/K]
    RH = 1.3*(1.e-9)*(Mwlin ** 0.574)   # [cm]
    etas = np.exp(-00.099 + (496/T)-1.5939 * np.log(T)) # [Pa.s]
    Dplin = kB*T/(6*3.1415*etas*RH)     # [m3/(s.cm)] = [cm2/s * 1.e-6]
    Dplin = Dplin*(1.e6)*60             # [cm2/min]


    wp = 1 - wm     # [ADM]
    gamma = 0.763   # [ADM]
    Vfm = 0.149 + 2.9*(1.e-4)*(T-273.15)
    Vfp = 0.0194 + 1.3*(1.e-4)*(T-273.15 - 105)
    Vf = wm*Vm_esp*Vfm + wp*Vp_esp*Vfp
    epsillon13 = Vi_esp*Mji/(Vp_esp*Mjp)    # [ADM]
    factor = -(gamma/epsillon13) * (wm*Vm_esp + wp*Vp_esp*epsillon13)/Vf - (1/Vfm)   # [ADM]
    
    Dp = Dplin*((Mwlin ** 2)/(Mw ** 2+1.e-20))*np.exp(factor)   # [cm2/min]

    
    Fseg = 0.12872374   # [ADM]
    Dpe = Fseg*Dp   # [cm2/min]

    invjc = 1/jc0 + 2*phip/Xc0  # [ADM]
    jc = 1/invjc                # [ADM]
    tau = np.sqrt(3/(2*jc*(delta ** 2)))    # [1/cm]
    rt = np.sqrt(np.log(1000*(tau ** 3)/(Na*lambda0*(3.1415 ** (1.5)))))/tau # [cm] (1000 converts L -> cm3)
    rm = rt # [cm]
    
    tauRt = 1/(ktc0*lambda0)    # [min]
    tauDt = (rm ** 2)/(3*Dpe)   # [min]
    
    ktc = ktc0/(1+tauDt/tauRt)  # [L/(mol.min)]
    
    A = 8*3.1415*(delta ** 3)*np.sqrt(jc)*Na/(3000)
    
    ktres = A*kp*M

    kte = ktc + ktres

    return kte         # [L/(mol.min)]

# ================================================================================== #
# ================================ REACTION FUNCTION =============================== #
# ================================================================================== #

Vantes = V0

def reac(t,Y):
    global Vantes, lambda0, Rcte, V0, Mjm, T

    # Monomer properties
    rhom = 0.968 - 1.225*(1.e-3)*(T-273.15) # [g/cm3] monomer density (T in oC)
    rhom = rhom*1000                       # [g/L] monomer density
    M0 = rhom/Mjm   # [mol/L] initial monomer concentration

    # Polymer properties
    e = 0.183+9*(1.e-4)*(T-273.15)  # [ADM] volume contraction factor
    rhop = rhom*(1+e)                # [g/L] monomer density

    # Current variables value 
    V = Y[0]
    M = Y[1]
    I = Y[2]
    mu0 = Y[3]
    mu1 = Y[4]
    mu2 = Y[5]
    X = (M0*V0-M*V)/(M0*V0)
    dV = Vantes - V

    if(V<0):
        V=0
    if(M<0):
        M=0
    if(I<0):
        I=0
    if(mu0 < 0):
        mu0 = 0
    if(mu1 < 0):
        mu1 = 0
    if(mu2 < 0):
        mu2 = 0
    if(mu0 == 1.e-20):
        lambda0 = 1.e-20

    Mw = mu2/(mu1+1.e-20)

    mm = M*V*rhom           # [g] monomer mass
    mm0 = M0*V0*rhom        # [g] monomer initial mass
    mp = mm0 - mm           # [g] polymer mass
    wm = mm/(mm+mp)         # [ADM]
    phip = (mp/rhop)/(mp/rhop + mm/rhom) # [ADM]
    
    # kinetic paremeters
    f = 0.58                                                # [ADM]
    kp = fkp(T,lambda0,wm,phip)                             # [L/(mol.min)]
    ktc = fktc(T,lambda0,phip,Mw,wm,kp,M)                   # [L/(mol.min)]
    kf = 4.66*(1.e9)*np.exp(-76290/(8.314*T))*60             # [L/(mol.min)]
    ktd = ktc*(3.956*(1.e-4)*np.exp(4090/(Rcte*T)))   # [L/(mol.min)]
    kd = 6.32*(1.e16*np.exp(-30660/(Rcte*T)))   # [1/min]
    kt = ktd + ktc
    
    #print('kp = %f ; ktc = %f ; kf = %f ; ktd = %f ; kd = %f'%(kp, ktc, kf, ktd, kd))
    
    

    # QSSA applied on R radical and lambda 0, 1 and 2
    ki_PR = 2*f*kd*I/(M+1.e-20)
    if(ki_PR<0):
        ki_PR = 0

    lambda0 = np.sqrt(ki_PR*M/kt)           # [mol/L] dlambda0

    lambda1 = (ki_PR*M + kp*M*lambda0 + kf*M*lambda0)/(kf*M + kt*lambda0)   # [mol/L] dlambda1
    
    lambda2 = (ki_PR*M + kp*M*(2*lambda1+lambda0) + kf*M*lambda0)/(kt*lambda0 + kf*M)
    #print(lambda0,mu2,mu1, mu2/(mu1+1.e-20))
    #os.system("PAUSE")
    #print(ktc,ktd,kt)
    # other constants
    eps = 1.e-20

    # Balance equations
    dy = np.zeros(6)   

    dy[0] = -V0*e*(ki_PR + (kp + kf)*lambda0)*(1-X) # [L] dV
    dy[1] = -ki_PR*M - (kp + kf)*M*lambda0 - (M/(V+eps))*dV         #[mol/L] dM
    dy[2] = -kd*I - (I/(V+eps))*dV # [mol/L] dI
    dy[3] = (ktd+0.5*ktc)*(lambda0 ** 2)+kf*M*lambda0 - (mu0/(V+eps))*dV # [mol/L] dmu0
    dy[4] = kt*lambda0*lambda1 + kf*M*lambda1 - (mu1/(V+eps))*dV # [mol/L] dmu1
    dy[5] = kt*lambda0*lambda2 + ktc*(lambda1 ** 2) + kf*M*lambda2 -(mu2/(V+eps))*dV#[mol/L] dmu2
    
    Vantes = V
    return dy

# ================================================================================== #
# =================================== INTEGRATION  ================================= #
# ================================================================================== #

YY = np.zeros(NInputVar,dtype=float)
YY = ode(reac).set_integrator('dopri5')
YY.set_initial_value(InputVar,t0)
Y = np.zeros((int(Nt),len(YY.y)),dtype=float)
dt = (tf-t0)/(Nt) #[min] integration interval
j=0

sys.stdout.write('\r'+'00%')
sys.stdout.flush()

while YY.successful() and YY.t < tf and j<Nt:
           
    Y[j,:] = YY.y[:]
    
    j = j + 1

    if(YY.t>10000):
        os.system("PAUSE")

    ctrl = int(YY.t/tf*100)
    sys.stdout.write('\r'+str(ctrl)+'%')
    sys.stdout.flush()

    YY.integrate(YY.t+dt)

sys.stdout.write('\r'+'100%')   
print ('\tFim integracao, t = '+str(YY.t))


# ================================================================================== #
# ==================================== RESULTS ===================================== #
# ================================================================================== #

Vt = Y[:,0] 
Mt = Y[:,1]
I = Y[:,2]
Mu0 = Y[:,3]
Mu1 = Y[:,4]
Mu2 = Y[:,5]

# PDI/X
X = np.zeros(Nt)
PDI = np.zeros(Nt)
Mn = np.zeros(Nt)
Mw = np.zeros(Nt)
for i in range(0,Nt):
    X[i] = (M0*V0-Mt[i]*Vt[i])/(M0*V0)
    Mn[i] = Mjm*Mu1[i]/(Mu0[i]+1.e-20)
    Mw[i] = Mjm*Mu2[i]/(Mu1[i]+1.e-20)
    PDI[i] = Mw[i]/(Mn[i]+1.e-20)

# ================================================================================== #
# ==================================== GRAPHS ====================================== #
# ================================================================================== #

plotgraphs(tArray,X,PDI,Mn,Mw)