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


def rtan(y,x):
    """a quadrant dependent tangents in radians!"""
    if np.all(np.isclose([x,y],[0.0,0.0])):
        return 0.0
    if np.isclose(x,0.0):
        if y<0.0:
            return -np.pi/2.0
        else:
            return np.pi/2.0
    if np.abs(y) < np.abs(x):
        val = np.arctan(np.abs(y/x))
        if x < 0.0:
            val = np.pi-val
        if y < 0.0:
            val = -val
        return val
    else:
        val = np.pi/2.0 - np.arctan(np.abs(x/y))
        if x < 0.0:
            val = np.pi - val
        if y < 0.0:
            val = -val
        return val


def Rtand(y,x):
    """a quadrant dependent tangents in degrees"""
    radians = rtan(y,x)
    return np.rad2deg(radians)


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

def test_rtan():
    assert(np.isclose(rtan(1.0,2.0),0.463647609))
    assert(np.isclose(rtan(1.0,-2.0),2.677945045))
    assert(np.isclose(rtan(-1.0,2.0),-0.463647609))
    assert(np.isclose(rtan(-1.0,-2.0),-2.677945045))
    assert(np.isclose(rtan(2.0,1.0),1.107148718))
    assert(np.isclose(rtan(2.0,-1.0),2.034443936))
    assert(np.isclose(rtan(-2.0,1.0),-1.107148718))
    assert(np.isclose(rtan(-2.0,-1.0),-2.034443936))
    assert(np.isclose(rtan(0.0,0.0),0.0))
    assert(np.isclose(rtan(1.0,0.0),np.pi/2.0))
    assert(np.isclose(rtan(-1.0,0.0),-np.pi/2.0))

    x,y = 10*(1-2*np.random.rand(2))
    radians = rtan(y,x)
    degrees = Rtand(y,x)
    assert(np.isclose(np.deg2rad(degrees),radians))


    





