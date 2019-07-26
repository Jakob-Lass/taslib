import numpy as np
from trigd import Cosd,Sind,Acosd
from functools import partial

def defaultValue(obj,value):
    """return object if not None else return value"""
    if obj is None:
        return value
    else:
        return obj

def directToReciprocalLattice(directLattice):
    """Calculate reciprocal lattice from direct lattice"""

    reciprocal = Cell()
    alfa = directLattice.alpha;
    beta = directLattice.beta;
    gamma = directLattice.gamma;

    cos_alfa = Cosd(alfa);
    cos_beta = Cosd(beta);
    cos_gamma = Cosd(gamma);

    sin_alfa = Sind(alfa);
    sin_beta = Sind(beta);
    sin_gamma = Sind(gamma);

    reciprocal.alpha =Acosd((cos_beta * cos_gamma - cos_alfa) / sin_beta / sin_gamma);
    reciprocal.beta = Acosd((cos_alfa * cos_gamma - cos_beta) / sin_alfa / sin_gamma);
    reciprocal.gamma =Acosd((cos_alfa * cos_beta - cos_gamma) / sin_alfa / sin_beta);

    ad = directLattice.a;
    bd = directLattice.b;
    cd = directLattice.c;

    arg = 1 + 2 * cos_alfa * cos_beta * cos_gamma - cos_alfa * cos_alfa -\
        cos_beta * cos_beta - cos_gamma * cos_gamma;
    if (arg < 0.0):
        raise AttributeError('Reciprocal lattice has negative volume!')
    

    vol = ad * bd * cd * np.sqrt(arg)/(2 * np.pi); # Added 2pi for new convention
    reciprocal.a = bd * cd * sin_alfa / vol;
    reciprocal.b = ad * cd * sin_beta / vol;
    reciprocal.c = bd * ad * sin_gamma / vol;

    return reciprocal


reciprocalToDirectLattice = directToReciprocalLattice
reciprocalToDirectLattice.__doc__ = "Calculate direct lattice from reciprocal lattice"

def calculateBMatrix(direct):
    """Calculate B matrix from lattice"""

    reciprocal = direct.directToReciprocalLattice()
    B = np.zeros((3,3))
    B[0,0] = reciprocal.a;
    B[0,1] = reciprocal.b * Cosd(reciprocal.gamma);
    B[0,2] = reciprocal.c * Cosd(reciprocal.beta);

    #    middle row
    
    B[1,1] = reciprocal.b * Sind(reciprocal.gamma);
    B[1,2] = -reciprocal.c * Sind(reciprocal.beta) * Cosd(direct.alpha);

    
    #    bottom row
    
    B[2,2] = 2 * np.pi / direct.c;

    return B

def cellFromUB(UB):
    GINV = np.einsum('ji,jk->ik',UB,UB)
    G = np.linalg.inv(GINV)*(2*np.pi)**2
    
    a = np.sqrt(G[0][0]);
    b = np.sqrt(G[1][1]);
    c = np.sqrt(G[2][2]);
    alpha = Acosd(G[1][2] / (b * c));
    beta = Acosd(G[2][0] / (a * c));
    gamma = Acosd(G[0][1] / (a * b)); # Change c -> b
    return Cell(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)


class Cell(object):
    """Cell object to hold information about crystal cell structures"""

    def __init__(self,a=None,b=None,c=None,alpha=None,beta=None,gamma=None, UB = None):
        """If a parameter is not specified, it is assumed to be 1.0 for lengths and 90 degrees for angles unless 3x3 matrix is given"""
        
        if UB is None:
            self.a = defaultValue(a,1.0)
            self.b = defaultValue(b,1.0)
            self.c = defaultValue(c,1.0)

            self.alpha = defaultValue(alpha,90.0)
            self.beta = defaultValue(beta,90.0)
            self.gamma = defaultValue(gamma,90.0)
        else:
            _cell = cellFromUB(UB)
            self.a,self.b,self.c,self.alpha,self.beta,self.gamma = \
                _cell.a,_cell.b,_cell.c,_cell.alpha,_cell.beta,_cell.gamma

    def __str__(self):
        returnString = 'cell.Cell('
        keyString = []
        for key in ['a','b','c','alpha','beta','gamma']:
            keyString.append('{:}={:.1f}'.format(key,getattr(self,key)))
        
        returnString += ', '.join(keyString)+')'
        return returnString

    def __eq__(self,other):
        keys = ['a','b','c','alpha','beta','gamma']
        selfValues = [getattr(self,key) for key in keys]
        otherValues = [getattr(other,key) for key in keys]
        return np.all(np.isclose(selfValues,otherValues))

    directToReciprocalLattice = directToReciprocalLattice
    reciprocalToDirectLattice = reciprocalToDirectLattice
    calculateBMatrix = calculateBMatrix


def test_Cell_error():
    cell = Cell(-10,-0.1,-0.2,-0.,-0.,-90)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore the warnings temporary
            reciprocal = directToReciprocalLattice(cell)
        assert False # pragma: no cover
    except AttributeError as e:
        assert True
        


def test_Cell_init():
    cell = Cell() # Default cell object

    for length in ['a','b','c']:
        assert(getattr(cell,length) == 1.0)
    for length in ['alpha','beta','gamma']:
        assert(getattr(cell,length) == 90.0)

    nonStandard = {'a':10,'b':11,'c':0.2,'alpha':60,'beta':90.1,'gamma':120}

    nonStandardCell = Cell(**nonStandard)

    for key,value in nonStandard.items():
        assert(getattr(nonStandardCell,key) == value)

    cellString = str(cell)
    wantedString = "cell.Cell(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0)"
    assert(cellString == wantedString)

def test_Cell_reciprocal():
    # The default cell is equal to its reciprocal with lattice vectors multiplied with 1/(2*pi)

    defaultCell = Cell()
    reciprocal = directToReciprocalLattice(defaultCell)
    for length in ['a','b','c']:
        assert(np.all(np.isclose(getattr(reciprocal,length), 2*np.pi)))
    for length in ['alpha','beta','gamma']:
        assert(np.all(np.isclose(getattr(reciprocal,length), 90.0)))
    
    # Back and forth is the same
    defaultCell2 = reciprocal.reciprocalToDirectLattice()

    assert defaultCell == defaultCell2


def test_Cell_BMatrix():

    # B matrix for 6.11, 6.11, 11.35, 90.0, 90.0, 120.0 as calculated by six
    BFromSix = np.array([[+0.188985358,+0.094492679,+0.000000000],
                         [+0.000000000,+0.163666121,+0.000000000],
                         [+0.000000000,+0.000000000,+0.088105727]])*2*np.pi # See note
                        
    lattice = Cell(6.11, 6.11, 11.35, 90.0, 90.0, 120.0)
    B = lattice.calculateBMatrix()
    print(B)
    print(BFromSix)
    assert(np.all(np.isclose(B,BFromSix,atol=1e-9)))

def test_Cell_CellFromUBMatrix():

    latticeCostants = 20*np.random.rand(3)+1.0
    angles = 90*np.random.rand(3)

    # To ensure possitive volumen
    a = Cosd(angles[0])
    b = Cosd(angles[1])

    lowerLimit = a*b - np.sqrt((a**2 - 1)*(b**2 - 1))
    upperLimit = np.sqrt((a**2 - 1)*(b**2 - 1)) + a*b
    angles[2] = Acosd(0.5*(lowerLimit+upperLimit))

    arguments = list(latticeCostants)+list(angles)
    
    lattice = Cell(*arguments)
    B = lattice.calculateBMatrix()
    
    lattice2 = Cell(UB=B);
    print(lattice)
    print(lattice2)
    assert(lattice==lattice2)
