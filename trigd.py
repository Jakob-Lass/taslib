import math
import numpy as np


def Sign(d):
    """Returns sign of provided array"""

    return np.sign(d)

def Sind(d):
    """Returns sine in degrees of array d"""

    return np.sin(np.deg2rad(d))

def Tand(d):
    """Returns tangent in degrees of array d"""

    return np.tan(np.deg2rad(d))

def Cosd(d):
    """Returns cosine in degrees of array d"""

    return np.cos(np.deg2rad(d))

def Cotd(d):
    """Returns cotangent in degrees of array d"""

    return np.reciprocal(np.tan(np.deg2rad(d)))

def Atand(d):
    """Returns argus tangents in degrees of array d"""

    return np.rad2deg(np.arctan(d))

def Atand2(x,y):
    """Returns argus tangents 2 in degrees of array x,y"""

    return np.rad2deg(np.arctan2(x,y))

def Acosd(d):
    """Returns argus cosine in degrees of array d"""

    return np.rad2deg(np.arccos(d))

def Asind(d):
    """Returns argus sine in degrees of array d"""

    return np.rad2deg(np.arcsin(d))



def test_Sign():
    values = 1-2*np.random.rand(20)
    assert(np.all(np.sign(values)==Sign(values)))

def test_Sind():
    values = 2*np.pi*(1-2*np.random.rand(200)) # values between -2 pi and 2 pi
    radian = np.sin(values)
    degree = Sind(values*180.0/np.pi)

    assert(np.all(np.isclose(radian,degree)))

def test_Cosd():
    values = 2*np.pi*(1-2*np.random.rand(200)) # values between -2 pi and 2 pi
    radian = np.cos(values)
    degree = Cosd(values*180.0/np.pi)

    assert(np.all(np.isclose(radian,degree)))

def test_Tand():
    values = 2*np.pi*(1-2*np.random.rand(200)) # values between -2 pi and 2 pi
    radian = np.tan(values)
    degree = Tand(values*180.0/np.pi)

    assert(np.all(np.isclose(radian,degree)))

def test_Cotd():
    values = 2*np.pi*(1-2*np.random.rand(200)) # values between -2 pi and 2 pi
    radian = np.reciprocal(np.tan(values))
    degree = Cotd(values*180.0/np.pi)

    assert(np.all(np.isclose(radian,degree)))

def test_Atand():
    values = 20000*(1-2*np.random.rand(200)) # values between -20000 and 20000

    radian = np.arctan(values)*180.0/np.pi
    degree = Atand(values)

    assert(np.all(np.isclose(radian,degree)))

def test_Atand2():
    x = 20000*(1-2*np.random.rand(200)) # values between -20000 and 20000
    y = 20000*(1-2*np.random.rand(200)) # values between -20000 and 20000

    radian = np.arctan2(x,y)*180.0/np.pi
    degree = Atand2(x,y)

    assert(np.all(np.isclose(radian,degree)))

def test_Asind():
    values = 1-2*np.random.rand(200) # values between -1 and 1

    radian = np.arcsin(values)*180.0/np.pi
    degree = Asind(values)

    assert(np.all(np.isclose(radian,degree)))

def test_Acosd():
    values = 1-2*np.random.rand(200) # values between -1 and 1

    radian = np.arccos(values)*180.0/np.pi
    degree = Acosd(values)

    assert(np.all(np.isclose(radian,degree)))