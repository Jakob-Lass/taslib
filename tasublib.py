import numpy as np
from collections import namedtuple
from trigd import Sind,Cosd,Rtand,Atand2,Acosd
from copy import deepcopy

def fmod(x,y):
    s = np.sign(x)
    res = s*np.mod(np.abs(x),y)
    return res


class tasQEPosition(object):
    def __init__(self,ki,kf,qh,qk,ql,qm):
        self.ki = ki
        self.kf = kf
        self.qh = qh
        self.qk = qk
        self.ql = ql
        self.qm = qm


class tasAngles(object):
    def __init__(self,monochromator_two_theta,a3,sample_two_theta,
                            sgl,sgu,analyzer_two_theta):
        self.monochromator_two_theta = monochromator_two_theta
        self.a3 = a3
        self.sample_two_theta = sample_two_theta
        self.sgu = sgu
        self.sgl = sgl
        self.analyzer_two_theta = analyzer_two_theta


class tasReflection(object):
    def __init__(self,qe=None,angles=None,ki=None,kf=None,qh=None,qk=None,ql=None,qm=None,
                        monochromator_two_theta = None, a3 = None,sample_two_theta = None,
                            sgl=None,sgu=None,analyzer_two_theta=None):
        if isinstance(qe,tasReflection): # Compy operator
            self.qe = deepcopy(qe.qe)
            self.angles = deepcopy(qe.angles)
        else:
            if qe is None:
                self.qe = tasQEPosition(ki,kf,qh,qk,ql,qm)
            else:
                self.qe = qe
            if angles is None:
                self.angles = tasAngles(monochromator_two_theta,a3,sample_two_theta,
                                sgl,sgu,analyzer_two_theta)
            else:
                self.angles = angles

    def __getattr__(self,key):
        #if key in ['qe','angles']: # Is automatically tested
        #    return self.__dict__[key]
        if key in self.qe.__dict__.keys():
            return getattr(self.qe,key)
        elif key in self.angles.__dict__.keys():
            return getattr(self.angles,key)
        else:
            raise AttributeError("'tasReflection' object hs no attribute '{}'".format(key))

ECONST = 2.072#2.072122396

def energyToK(energy):
    """Convert energy in meV to K in q/A"""
    return np.sqrt(energy / ECONST)

def KToEnergy(K):
    """Convert K in 1/A to E in meV"""
    return ECONST*np.power(K,2.0)


def tasReflectionToHC(r, B):
    """Calculate HC from HKL and B matrix"""
    return tasHKLToHC(r.qh,r.qk,r.ql,B)

def tasHKLToHC(qh,qk,ql, B):
    """Calculate HC from reflection r and B matrix"""
    h = np.array([qh,qk,ql])
    hc = np.dot(B,h)
    return hc

def calcTheta(ki, kf, two_theta):
    """
                  |ki| - |kf|cos(two_theta)
    tan(theta) = --------------------------
                   |kf|sin(two_theta)
    """
    return Rtand(np.abs(ki) - np.abs(kf) * Cosd(two_theta),
              np.abs(kf) * Sind(two_theta))

def tasAngleBetween(v1, v2):
    angle = np.dot(v1, v2) / (np.linalg.norm(v1,axis=0) * np.linalg.norm(v2,axis=0));
    v3 = np.cross(v1, v2)

    angles = np.linalg.norm(v3,axis=0) / (np.linalg.norm(v1,axis=0) * np.linalg.norm(v2,axis=0));
    angle = Atand2(angles, angle);
    #sum = np.sum(v3,axis=0)
    #if sum<0.0:
    #    angle *= -1.
    
    return angle;


def tasAngleBetweenReflections(B, r1, r2):
    """Calculate angle between two reflections"""
    return tasAngleBetweenReflectionsHKL(B,r1.qh,r1.qk,r1.ql,r2.qh,r2.qk,r2.ql)
    

def tasAngleBetweenReflectionsHKL(B,h1,k1,l1,h2,k2,l2):
    """Calculate angle between two reflections"""
    v1 = np.array([h1,k1,l1])
    v2 = np.array([h2,k2,l2])

    chi1 = np.einsum('ij,j...->i...',B,v1)
    chi2 = np.einsum('ij,j...->i...',B,v2)

    angle = tasAngleBetween(chi1,chi2)
    return angle

def uFromAngles(om,sgu,sgl):
    u = np.array([Cosd(om)*Cosd(sgl),
                  -Sind(om)*Cosd(sgu)+Cosd(om)*Sind(sgl)*Sind(sgu),
                  Sind(om)*Sind(sgu)+Cosd(om)*Sind(sgl)*Cosd(sgu)])
    return u

def calcTasUVectorFromAngles(rr):
    ss = np.sign(rr.sample_two_theta)

    r = tasReflection(rr)
    r.sample_two_theta = np.abs(r.sample_two_theta)
    theta = calcTheta(r.ki,r.kf,r.sample_two_theta)

    om = r.angles.a3 - ss*theta
    m = uFromAngles(om, r.angles.sgu, ss*r.angles.sgl); 
    return m


def tasReflectionToQC(r, UB):
    return tasReflectionToQCHKL(r.qh,r.qk,r.ql,UB)

def tasReflectionToQCHKL(h,k,l,UB):
    Q = np.array([h,k,l])
    return np.einsum('ij,j...->i...',UB,Q)


def makeAuxReflection(B, r1, ss, hkl):
    r2 = tasReflection(r1)
    r2.qe.qh,r2.qe.qk,r2.qe.ql = hkl

    theta = calcTheta(r1.qe.ki, r1.qe.kf, 
        ss*r1.angles.sample_two_theta);
    om = r1.angles.a3 - ss*theta;
    
    om += tasAngleBetweenReflectionsHKL(B,r1.qh,r1.qk,r1.ql,*hkl)

    QC = tasReflectionToHC(r2.qe, B)
    
    q = np.linalg.norm(QC)
    
    cos2t = np.divide(r1.ki * r1.ki + r1.kf * r1.kf - q * q, \
      (2. * np.abs(r1.ki) * np.abs(r1.kf)))
    if np.abs(cos2t) > 1. :
        raise RuntimeError('Scattering angle not closed!') # pragma: no cover
    
    r2.angles.sample_two_theta = ss * Acosd(cos2t);
    theta = calcTheta(r1.qe.ki, r1.qe.kf, ss*r2.angles.sample_two_theta);
    r2.angles.a3 = om + ss*theta;

    r2.angles.a3 = fmod(r2.angles.a3 + ss*180.,360.) - ss*180.;
    
    return r2 

def calcTwoTheta(B,ref,ss):
    QC = tasReflectionToHC(ref, B);

    q = np.linalg.norm(QC);

    cos2t = np.divide(ref.ki * ref.ki + ref.kf * ref.kf - q * q, \
        (2. * np.abs(ref.ki) * np.abs(ref.kf)))
    
    if (np.abs(cos2t) > 1.):
        raise RuntimeError('Calculated abs(cos2t) value {} bigger than 1! Scattering angle not closed'.format(np.abs(cos2t))) # pragma: no cover

    value = ss * Acosd(cos2t);
    return value


def calcPlaneNormal(r1, r2):
    u1 = calcTasUVectorFromAngles(r1)
    u2 = calcTasUVectorFromAngles(r2)
    planeNormal = np.cross(u1, u2)
    planeNormal *= 1.0/np.linalg.norm(planeNormal)

    # In TasCode code is commented out performing check for sign of planeNormal[2] is performed.
    # If negative, z component negated.
    
    planeNormal[2] = np.abs(planeNormal[2])
    return planeNormal;

def matFromTwoVectors(v1,v2):
    a1 = v1/np.linalg.norm(v1)
    
    a3 = np.cross(a1,v2)
    a3 *= 1.0/np.linalg.norm(a3)

    a2 = np.cross(a1,a3)

    result = np.zeros((3,3))
    for i in range(3):
        result[i][0] = a1[i]
        result[i][1] = a2[i]
        result[i][2] = a3[i]

    return result




def calcTasUBFromTwoReflections(cell,r1,r2,):

    B = cell.calculateBMatrix();

    h1 = tasReflectionToHC(r1.qe, B);
    h2 = tasReflectionToHC(r2.qe, B);
    
    HT = matFromTwoVectors(h1, h2);
    
    #   calculate U vectors and UT matrix

    u1 = calcTasUVectorFromAngles(r1);
    u2 = calcTasUVectorFromAngles(r2);
    
    UT = matFromTwoVectors(u1, u2);

    #   UT = U * HT

    U = np.dot(UT, HT.T);

    UB = np.dot(U, B);

    return UB;

def buildRMatrix(UB,planeNormal,qe):
    U1V = tasReflectionToQC(qe, UB);

    U1V*=1.0/np.linalg.norm(U1V);


    U2V = np.cross(planeNormal, U1V); 

    if (np.linalg.norm(U2V) < .0001):
        raise RuntimeError('Found vector is too short') # pragma: no cover
    TV = buildTVMatrix(U1V, U2V);

    TVINV = np.linalg.inv(TV);
    return TVINV;

def buildTVMatrix(U1V, U2V):
    U2V*=1.0/np.linalg.norm(U2V);
    T3V = np.cross(U1V, U2V);
    T3V *= 1.0/np.linalg.norm(T3V)

    T = np.zeros((3,3))

    for i in range(3):
        T[i][0] = U1V[i];
        T[i][1] = U2V[i];
        T[i][2] = T3V[i];

    return T;


def calcTasQAngles(UB, planeNormal, ss, a3offset, qe):

    R = buildRMatrix(UB, planeNormal, qe);
    angles = tasAngles(0,0,0,0,0,0)

    cossgl = np.sqrt(R[0][0]*R[0][0]+R[1][0]*R[1][0]);
    angles.sgl = ss*Atand2(-R[2][0],cossgl);
    if (np.abs(angles.sgl - 90.) < .5):
        raise RuntimeError('Combination of UB and Q is not valid') # pragma: no cover
    
    #    Now, this is slightly different then in the publication by M. Lumsden.
    #    The reason is that the atan2 helps to determine the sign of om
    #    whereas the sin, cos formula given by M. Lumsden yield ambiguous signs 
    #    especially for om.
    #    sgu = atan(R[2][1],R[2][2]) where:
    #    R[2][1] = cos(sgl)sin(sgu)
    #    R[2][2] = cos(sgu)cos(sgl)
    #    om = atan(R[1][0],R[0][0]) where:
    #    R[1][0] = sin(om)cos(sgl)
    #    R[0][0] = cos(om)cos(sgl)
    #    The definitions of the R components are taken from M. Lumsden
    #    R-matrix definition.
    

    om = Atand2(R[1][0]/cossgl, R[0][0]/cossgl);
    angles.sgu = Atand2(R[2][1]/cossgl, R[2][2]/cossgl);

    QC = tasReflectionToQC(qe, UB);
    
    q = np.linalg.norm(QC);
    
    cos2t = (qe.ki * qe.ki + qe.kf * qe.kf -\
        q * q) / (2. * np.abs(qe.ki) * np.abs(qe.kf));
    if (np.abs(cos2t) > 1.):
        raise RuntimeError('Scattering angle cannot be closed, cos2t = ',cos2t) # pragma: no cover
    theta = calcTheta(qe.ki, qe.kf, Acosd(cos2t));
    angles.sample_two_theta = ss * Acosd(cos2t);


    angles.a3 = om + ss*theta + a3offset;
    #
    #    put a3 into -180, 180 properly. We can always turn by 180 because the
    #    scattering geometry is symmetric in this respect. It is like looking at
    #    the scattering plane from the other side
    
    angles.a3 = fmod(angles.a3 + ss*180.,360.) - ss*180.; 
    return angles


def calcScatteringPlaneNormal(qe1,qe2):
  
    v1 = [qe1.qh,qe1.qk,qe1.ql];
    v2 = [qe2.qh,qe2.qk,qe2.ql];

    planeNormal = np.cross(v1, v2);
    planeNormal *= 1.0/np.linalg.norm(planeNormal)
    
    return planeNormal;
    

###################################################### tests ##############################
def test_energyToK():
    energy = 5.00
    K = energyToK(energy)
    print(K)
    assert(np.isclose(K,1.553424415003))#1.5533785355359282))
    
    energy2 = KToEnergy(K)
    assert(np.isclose(energy2,energy))


def test_tasReflectionToHC():
    from cell import Cell

    lattice = Cell(12.32,3.32,9.8,93,45,120)
    B = lattice.calculateBMatrix()
    energy = 5.00
    K = energyToK(energy)

    reflection = tasReflection(
        ki=K, kf=K, qh=2, qk=0, ql=-1)
    hc = tasReflectionToHC(reflection, B)

    result = np.array([+0.434561180,-0.005347733,-0.102040816]) * 2*np.pi # From Six
    
    assert(np.all(np.isclose(hc, result)))


    
def test_tasReflection_initialization():


    tR = tasReflection(ki=1.5,kf=2.5,qh=1,qk=2,ql=3,qm=2.0,
                        monochromator_two_theta = 72, a3 = 0.0,sample_two_theta = 60.6,
                            sgl=0.1,sgu=-0.1,analyzer_two_theta=50.0)
    
    for key,val in {'ki':1.5,'kf':2.5,'qh':1,'qk':2,'ql':3,'qm':2.0,
                        'monochromator_two_theta':72, 'a3' : 0.0,'sample_two_theta' : 60.6,
                            'sgl':0.1,'sgu':-0.1,'analyzer_two_theta':50.0}.items():
        assert(np.isclose(getattr(tR,key),val))

    qe = tasQEPosition(1.25,2.0,-3,2,-0,1.2)
    angles = tR.angles

    tR2 = tasReflection(qe,angles)

    angles2 = tR2.angles
    qe2 = tR2.qe
    for key in ['monochromator_two_theta', 'a3','sample_two_theta',
                            'sgl','sgu','analyzer_two_theta']:
        assert(np.isclose(getattr(angles2,key),getattr(angles,key)))
    
    for key,val in {'ki':1.25,'kf':2.0,'qh':-3,'qk':2,'ql':-0,'qm':1.2}.items():
        assert(np.isclose(getattr(qe,key),val))
        assert(np.isclose(getattr(qe,key),getattr(qe2,key)))

    try:
        tR2.notExisiting
        assert(False) # pragma: no cover
    except AttributeError as e:
        assert True


def test_calcTheta():
    ## in the case of elastic scattering, theta = 0.5*2theta

    ki = energyToK(5.0)
    kf = ki
    tTheta = 90.0
    theta = calcTheta(ki,kf,tTheta)
    assert(np.isclose(theta,0.5*tTheta))

    ThetaSix = 61.969840817
    ki = 2.0
    kf = 1.0
    tTheta = 42.0
    assert(np.isclose(calcTheta(ki=ki,kf=kf,two_theta=tTheta),ThetaSix))

def test_tasAngleBetween():
    v1 = np.array([1,0,0])
    v2 = np.array([0,1,1])
    v3 = np.array([1,0,1])

    assert(np.isclose(tasAngleBetween(v1,v1),0.0))
    assert(np.isclose(tasAngleBetween(v1,v2),90.0))
    assert(np.isclose(tasAngleBetween(v2,v1),90.0))
    assert(np.isclose(tasAngleBetween(v1,v3),45.0))
    assert(np.isclose(tasAngleBetween(v3,v1),45.0))
    assert(np.isclose(tasAngleBetween(v2,v3),60.0))
    assert(np.isclose(tasAngleBetween(v3,v2),60.0))
    


def test_tasAngleBetweenReflections():

    from cell import Cell
    lattice = Cell(6.11, 6.11, 11.35, 90.0, 90.0, 120.0)
    B = lattice.calculateBMatrix()

    r1 = tasReflection(qh=1.0,qk=-2.0,ql=0.0)
    r2 = tasReflection(qh=-1.5,qk=1.1,ql=1.0)
    
    angle = tasAngleBetweenReflections(B,r1,r2)
    assert(np.isclose(angle,131.993879212))

def test_uFromAngles():

    res1 = np.array([1.000000000,0.000000000,0.000000000])
    res2 = np.array([0.743144825,-0.669130606,0.000000000])
    res3 = np.array([0.960176274,-0.223387216,0.167808444])
    res4 = np.array([+0.974425454,-0.220358460,-0.044013456])
    res5 = np.array([0.974425454,0.220358460,0.044013456])
    res6 = np.array([0.974425454,0.220358460,-0.044013456])

    results = [res1,res2,res3,res4,res5,res6]

    params = [[0.0,0.0,0.0],[42.0,0.0,0.0],[12.0,-5.0,11.0],
              [12.0,11.0,-5.0],[-12.0,11.0,5.0],[-12.0,-11.0,-5.0]]
    
    for par,res in zip(params,results):
        print(par,res)
        assert(np.all(np.isclose(uFromAngles(*par),res)))

def test_calcTasUVectorFromAngles():
    tR = tasReflection(ki=1.5,kf=2.5,qh=1,qk=2,ql=3,qm=2.0,
                        monochromator_two_theta = 72, a3 = 10.0,sample_two_theta = -60.6,
                            sgu=0.1,sgl=-0.1,analyzer_two_theta=50.0)

    assert(np.all(np.isclose(calcTasUVectorFromAngles(tR),np.array([+0.955598330,-0.294664334,+0.002182125]))))


def test_tasReflectionToQC():
    r = tasQEPosition(ki=1.5,kf=2.5,qh=1,qk=2,ql=3,qm=2.0)
    from cell import Cell
    lattice = Cell(6.11, 6.11, 11.35, 90.0, 90.0, 120.0)
    B = lattice.calculateBMatrix()

    Q = tasReflectionToQC(r,B)
    assert(np.all(np.isclose(Q,np.array([+0.377970716,+0.327332242,+0.264317181])*2*np.pi))) # 2*pi for convention change

def test_calcTwoTheta():
    from cell import Cell
    lattice = Cell(6.11, 6.11, 11.35, 90.0, 90.0, 120.0)
    B = lattice.calculateBMatrix()

    hkl = [[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1],[0,0,-1],[2,-1,3]]
    result = [0.0,44.93975435373514,44.93975435373514,44.93975435373514,44.93975435373514,20.52777682289506,20.52777682289506,116.61092637820377]
    for (h,k,l),res in zip(hkl,result):
    
        qm = np.linalg.norm(np.dot(B,[h,k,l]))
        r1 = tasReflection(ki=1.553424,kf=1.553424,qh=h,qk=k,ql=l,qm=qm,monochromator_two_theta=74.2,a3=0.0,sample_two_theta=-200,sgu=0.0,sgl=0.0,analyzer_two_theta=74.2)
        
        tt = calcTwoTheta(B,r1,1)
        
        if not np.isclose(tt,res):
            print('Two theta for ({},{},{})={} but expected {}'.format(h,k,l,tt,res)) # pragma: no cover
            assert(False) # pragma: no cover

    try:
        h,k,l = 10,-10,10
        qm = np.linalg.norm(np.dot(B,[h,k,l])) # HKL is far out of reach
        r1 = tasReflection(ki=1.553424,kf=1.553424,qh=h,qk=k,ql=l,qm=qm,monochromator_two_theta=74.2,a3=0.0,sample_two_theta=-200,sgu=0.0,sgl=0.0,analyzer_two_theta=74.2)
        tt = calcTwoTheta(B,r1,1)
        assert False # pragma: no cover
    except RuntimeError:
        assert True
    

def test_matFromTwoVectors():
    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])

    assert(np.all(np.isclose(matFromTwoVectors(v1,v2),np.diag([1,-1,1]))))

    v1 = np.array([0,1,0])
    v2 = np.array([0,0,1])
    result = np.array([[0,0,1],[1,0,0],[0,-1,0]])
    
    assert(np.all(np.isclose(matFromTwoVectors(v1,v2),result)))

    v1 = np.array([0,1,2])
    v2 = np.array([0,2,1])
    result = np.array([[0,0,-1],[0.4472136 ,  -0.89442719,0],[0.89442719,  0.4472136, 0]])
    assert(np.all(np.isclose(matFromTwoVectors(v1,v2),result)))

def test_calcTasUBFromTwoReflections():
    
    from cell import Cell
    lattices = []
    r1s = []
    r2s = []
    UBs = []

    # YMnO3, Eu
    latticeYMnO3 = Cell(6.11, 6.11, 11.35, 90.0, 90.0, 120.0)
    Ei = 5.0
    ki = energyToK(Ei)
    Ef = 4.96
    kf = energyToK(Ef)

    r1 = tasReflection(ki=ki,kf=kf,qh=0,qk=-1,ql=0,a3=-60,sample_two_theta=-45.23,sgl=0,sgu=0)
    r2 = tasReflection(ki=ki,kf=kf,qh=1,qk=0,ql=0,a3=0,sample_two_theta=-45.23,sgl=0,sgu=0)
    
    

    UB = np.array([[0.023387708,-0.150714153,0.0],
    [-0.187532612,-0.114020655,-0.0],
    [0.0,-0.0,-0.088105727]])

    lattices.append(latticeYMnO3)
    r1s.append(r1)
    r2s.append(r2)
    UBs.append(UB)


    # PbTi
    latticePbTi = Cell(9.556, 9.556, 7.014, 90.0, 90.0, 90.0)
    Ei = 5.0000076

    ki = energyToK(Ei)
    Ef = 4.924756

    kf = energyToK(Ef)

    r1 = tasReflection(ki=ki,kf=kf,qh=0,qk=0,ql=2,a3=74.2,sample_two_theta=-71.1594,sgl=0,sgu=0)
    r2 = tasReflection(ki=ki,kf=kf,qh=1,qk=1,ql=2,a3=41.33997,sample_two_theta=-81.6363,sgl=0,sgu=0)
   

    UB = np.array([[0.06949673,0.0694967,-0.048957292],
                [-0.025409263,-0.025409255,-0.13390279],
                [-0.07399609,0.07399613,-4.2151984E-9]]) 

    lattices.append(latticePbTi)
    r1s.append(r1)
    r2s.append(r2)
    UBs.append(UB)

# SecuO3
    latticeSeCuo = Cell(7.725,8.241,8.502,90.0,99.16,90.0)
    Ei = 5.0000076

    ki = energyToK(Ei)
    Ef = 4.9221945
    kf = energyToK(Ef)

    r1 = tasReflection(ki=ki,kf=kf,qh=2,qk=1,ql=0,a3=47.841908,sample_two_theta=-72.0,sgl=0,sgu=0)
    r2 = tasReflection(ki=ki,kf=kf,qh=2,qk=-1,ql=0,a3=-1.8000551,sample_two_theta=-72.0,sgl=0,sgu=0)
   

    UB = np.array([[0.066903256,-0.10436039,0.009677114],
                [-0.11276933,-0.061914437,-0.016311338],
                [-0.0,0.0,-0.11761938]]) 

    lattices.append(latticeSeCuo)
    r1s.append(r1)
    r2s.append(r2)
    UBs.append(UB)


    for lat,r1,r2,UBSix in zip(lattices,r1s,r2s,UBs):    
        UB = calcTasUBFromTwoReflections(lat,r1,r2)
        assert(np.all(np.isclose(UB,UBSix*2*np.pi,atol=1e-6)))
    




def test_calcPlaneNormal():
    from cell import Cell
    lattice = Cell(6.11, 6.11, 11.35, 90.0, 90.0, 120.0)
    B = lattice.calculateBMatrix()

    HKL1 = [[1,0,0],[1,1,0],[1,0,0]]
    HKL2 = [[0,1,0],[1,-1,0],[0,0,1]]
    result=[[0,0,1],[0,0,1],[0,0,1]]
    
    for hkl1,hkl2,res in zip(HKL1,HKL2,result):

        qm = np.linalg.norm(np.dot(B,hkl1))
        r1 = tasReflection(ki=1.553424,kf=1.553424,qh=hkl1[0],qk=hkl1[1],ql=hkl1[2],qm=qm,monochromator_two_theta=74.2,a3=0.0,sample_two_theta=-200,sgu=0.0,sgl=0.0,analyzer_two_theta=74.2)
        tt = calcTwoTheta(B,r1,1)
        r1.angles.sample_two_theta = tt
        r1.angles.a3 = calcTheta(r1.ki,r1.kf,tt)
        print('r1.a3: ',r1.a3)

        qm = np.linalg.norm(np.dot(B,hkl2))
        r2 = tasReflection(ki=1.553424,kf=1.553424,qh=hkl2[0],qk=hkl2[1],ql=hkl2[2],qm=qm,monochromator_two_theta=74.2,a3=0.0,sample_two_theta=-200,sgu=0.0,sgl=0.0,analyzer_two_theta=74.2)
        tt = calcTwoTheta(B,r2,1)
        r2.angles.sample_two_theta = tt
        r2.angles.a3 = calcTheta(r2.ki,r2.kf,tt)+tasAngleBetweenReflections(B,r1,r2)
        print('r2.a3: ',r2.a3)

        planeNormal = calcPlaneNormal(r1,r2)
        if not np.all(np.isclose(planeNormal,res)):
            print('Plane normal for {} and {} = {} but expected {}'.format(hkl1,hkl2,planeNormal,res)) # pragma: no cover
            assert False # pragma: no cover

def test_makeAuxReflection():
    from cell import Cell
    lattice = Cell(12.32,3.32,9.8,92,91,120)
    B = lattice.calculateBMatrix()

    E = 22;
    ki = energyToK(E)
    h,k,l = 1,0,-1
    qm = np.linalg.norm(np.dot(B,[h,k,l]))

    
    r1 = tasReflection(ki=ki,kf=ki,qh=h,qk=k,ql=l,qm=qm,monochromator_two_theta=74.2,a3=12.2,sgu=0.0,sgl=0.0,analyzer_two_theta=74.2)
    
    tt = calcTwoTheta(B,r1,-1)
    r1.angles.sample_two_theta = tt
    
    
    r2 = makeAuxReflection(B,r1,-1.0,[3,1,-1])
    
    
    assert(np.isclose(r2.a3,35.875022341))
    assert(np.isclose(r2.sample_two_theta,-64.129182823))
    
    try:
        r2 = makeAuxReflection(B,r1,-1.0,[30,1,-1]) # HKL is far out of reach
        assert False # pragma: no cover
    except RuntimeError:
        assert True


def test_calcTasQAngles():
    from cell import Cell

    # SeCuO3
    latticeSeCuo = Cell(7.725,8.241,8.502,90.0,99.16,90.0)
    Ei = 5.0000076

    ki = energyToK(Ei)
    Ef = 4.9221945
    kf = energyToK(Ef)

    r1 = tasReflection(ki=ki,kf=kf,qh=2,qk=1,ql=0,a3=47.841908,sample_two_theta=-71.840689064,sgl=0,sgu=0)
    r2 = tasReflection(ki=ki,kf=kf,qh=2,qk=-1,ql=0,a3=-1.819579944281713,sample_two_theta=-71.840689064,sgl=0,sgu=0)
    UB = calcTasUBFromTwoReflections(latticeSeCuo,r1,r2)
    planeNormal = calcPlaneNormal(r1,r2)
    
    ss = -1
    a3Off = 0.0
    a3s = []
    for (h,k,l),a3,a4 in zip([[r1.qh,r1.qk,r1.ql],[r2.qh,r2.qk,r2.ql]],[r1.a3,r2.a3],[r1.sample_two_theta,r2.sample_two_theta]):
        
        R = tasReflection(ki=ki,kf=kf,qh=h,qk=k,ql=l)
        qm = np.linalg.norm(np.dot(UB,[h,k,l]))
        
        qe = tasQEPosition(ki,kf,R.qh,R.qk,R.ql,qm=qm)
        angles = calcTasQAngles(UB,planeNormal,ss,a3Off,qe)

        assert(np.isclose(a3,angles.a3))
        assert(np.isclose(a4,angles.sample_two_theta))
    

def test_calcScatteringPlaneNormal():
    
    qe1 = tasQEPosition(1.0,1.0,qh=1.0,qk=2.0,ql=3.0,qm=2)
    qe2 = tasQEPosition(1.0,1.0,qh=2.0,qk=-2.0,ql=3.0,qm=2)
    planeNormal = calcScatteringPlaneNormal(qe1,qe2)
    print(planeNormal)
    assert(np.isclose(np.linalg.norm(planeNormal),1.0))
    assert(np.all(np.isclose([0.87287156,0.21821789,-0.43643578],planeNormal)))
