import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, cpu_count
import matplotlib.pyplot as plt
import time
import os

WEIGHT = 1e-4       # critical weight for roulette
CHANCE = 0.1		    # Chance of roulette survival
PARTIALREFLECTION = 0     # 1=split photon, 0=statistical reflection.
COSZERO = 1.0 - 1.0e-12     # cosine of about 1e-6 rad
COS90D = 1.0e-6     # cosine of about 1.57 - 1e-6 rad
path = os.path.dirname(__file__)
modelfile = np.genfromtxt(os.path.join(path,'model_input.txt'), dtype=['<U20', np.float])

class Medium:
    """Medium class - optical medium class defining the optical properties
        Class instance variables:
            n - refractive index
            mua - absorption coefficient. [1/cm]
            mus - scattering coefficient. [1/cm]
            g - anisotropy
        Methods:

    """

    def __init__(self, mediumName, nummua):
        wavelength = np.loadtxt(os.path.join(path,'wavelength.csv'))
        water = np.loadtxt(os.path.join(path,'mua_water.csv'))
        melanin = np.loadtxt(os.path.join(path,'mua_melanin.csv'))
        deoxy = np.loadtxt(os.path.join(path,'mua_deoxy.csv'))
        oxy = np.loadtxt(os.path.join(path,'mua_oxy.csv'))
        # 修正
        water = water*10
        deoxy = deoxy*0.15
        oxy = oxy*0.15
        melanin = melanin*0.1
        # Rayleigh
        Rayleigh = (np.power(wavelength,-3.255))*7.84e+8
        # t3~t7 contain hemoglobin 
        S = 0.75
        gamma = 0.25*0.99*0.45
        # 定義各組織光學特性
        if mediumName.lower() == 'AIR'.lower():
            self.n = 1.0
            self.mua = np.zeros(316)[nummua]
            self.mus = 0.0
            self.g = 1.0
        elif mediumName.lower() == 'TISSUE1'.lower():
            Cwater = 0.05
            tissue1 = ((0.1-(0.3e-4*wavelength))+1.25*Rayleigh)*(1-Cwater)+Cwater*water
            self.n = 1.5
            self.mua = tissue1[nummua]
            self.mus = 1000.0
            self.g = 0.86
        elif mediumName.lower() == 'TISSUE2'.lower():
            Cwater = 0.2
            Cmel = 0.13
            tissue2 = (Cmel*melanin+(1-Cmel)*Rayleigh)*(1-Cwater)+Cwater*water
            self.n = 1.34
            self.mua = tissue2[nummua]
            self.mus = 450.0
            self.g = 0.80
        elif mediumName.lower() == 'TISSUE3'.lower():
            Cb = 0.04
            Cwater = 0.5
            tissue3 = ((1-S)*gamma*Cb*deoxy)+(S*gamma*Cb*oxy)+((1-gamma*Cb)*Cwater*water)+((1-gamma*Cb)*(1-Cwater)*Rayleigh)
            self.n = 1.4
            self.mua = tissue3[nummua]
            self.mus = 300.0
            self.g = 0.90
        elif mediumName.lower() == 'TISSUE4'.lower():
            Cb = 0.3
            Cwater = 0.6
            tissue4 = ((1-S)*gamma*Cb*deoxy)+(S*gamma*Cb*oxy)+((1-gamma*Cb)*Cwater*water)+((1-gamma*Cb)*(1-Cwater)*Rayleigh)
            self.n = 1.39
            self.mua = tissue4[nummua]
            self.mus = 350.0
            self.g = 0.95
        elif mediumName.lower() == 'TISSUE5'.lower():
            Cb = 0.04
            Cwater = 0.7
            tissue5 = ((1-S)*gamma*Cb*deoxy)+(S*gamma*Cb*oxy)+((1-gamma*Cb)*Cwater*water)+((1-gamma*Cb)*(1-Cwater)*Rayleigh)
            self.n = 1.4
            self.mua = tissue5[nummua]
            self.mus = 250.0
            self.g = 0.80
        elif mediumName.lower() == 'TISSUE6'.lower():
            Cb = 0.1
            Cwater = 0.7
            tissue6 = ((1-S)*gamma*Cb*deoxy)+(S*gamma*Cb*oxy)+((1-gamma*Cb)*Cwater*water)+((1-gamma*Cb)*(1-Cwater)*Rayleigh)
            self.n = 1.38
            self.mua = tissue6[nummua]
            self.mus = 300.0
            self.g = 0.95
        elif mediumName.lower() == 'TISSUE7'.lower():
            Cb = 0.05
            Cwater = 0.7
            tissue7 = ((1-S)*gamma*Cb*deoxy)+(S*gamma*Cb*oxy)+((1-gamma*Cb)*Cwater*water)+((1-gamma*Cb)*(1-Cwater)*Rayleigh)    
            self.n = 1.44
            self.mua = tissue7[nummua]
            self.mus = 50.0
            self.g = 0.75
               
class LayerStruct:
    """LayerStruct class - multi-layered structure
        Class instance variables:
            nIn - refractive index of the incidence medium
            nOut - refractive index of the exit medium
            numLayers - number of layers
            layer - list of layer objects
            layerThickness - layer thickness in [cm]
            layerZ - layer depth z coordinates, list表示[top bottom] [cm]
            cosCrit - ciritical angle cosines of each layer, list表示[top bottom]
        Methods:
            
    """
    
    def __init__(self, nummua=0):
        self.numLayers = int(modelfile[4].__getitem__(1))
        # 定義組織層(此例為上下兩層空氣，中間七層組織)
        self.layer = [Medium('AIR', nummua), \
                      Medium('TISSUE1', nummua), \
                      Medium('TISSUE2', nummua), \
                      Medium('TISSUE3', nummua), \
                      Medium('TISSUE4', nummua), \
                      Medium('TISSUE5', nummua), \
                      Medium('TISSUE6', nummua), \
                      Medium('TISSUE7', nummua), \
                      Medium('AIR', nummua)]
        self.layerThickness = np.empty([self.numLayers])
        for i in range (self.numLayers):
            self.layerThickness[i] = modelfile[5+i].__getitem__(1)
        # [20e-4, 80e-4, 150e-4, 80e-4, 1500e-4, 80e-4, 6000e-4]    
        # 不包括上下兩層空氣層
        # 總共8000e-4 [cm]
        self.layerZ = []
        self.cosCrit = []
        z = 0   # incidence first medium z coordinate [cm]
        self.layerZ.append([0, 0])  # first incidence medium, not used
        self.cosCrit.append([0, 0])  # first incidence medium, not used
        # find the z depth coordinates and cosine critical angles for each
        #   layer
        for i in range(1, self.numLayers+1):
            print('layer'+str(i)+'-->done!')
            self.layerZ.append([z, z+self.layerThickness[i-1]])
            z = self.layerZ[-1][1] #取出最後一個值，但此值還是列表，取該列表的第二個值
                                   #也就是最後一層layer底部的z點座標
            # calculate the critical angle cosines for each layer
            # crticial angle at top interface of the current layer
            n1 = self.layer[i].n
            n2 = self.layer[i-1].n
            if n1 > n2:
                cosCrit0 = (1.0 - n2*n2/(n1*n1))**0.5
            else:
                cosCrit0 = 0.0
            # crticial angle at bottom interface of the current layer
            n2 = self.layer[i+1].n
            if n1 > n2:
                cosCrit1 = (1.0 - n2*n2/(n1*n1))**0.5
            else:
                cosCrit1 = 0.0
            self.cosCrit.append([cosCrit0, cosCrit1])
            

    def calc_r_specular(self):
        # direct reflections from the 1st and 2nd layers. 鏡面反射
        temp = (self.layer[0].n - self.layer[1].n)/(self.layer[0].n + \
            self.layer[1].n)
        r1 = temp*temp
  
        if ((self.layer[1].mua == 0.0) and (self.layer[1].mus == 0.0)):
            # glass layer.
            temp = (self.layer[1].n - self.layer[2].n)/(self.layer[1].n + \
                self.layer[2].n)
            r2 = temp*temp
            r1 = r1 + (1 - r1)*(1 - r1)*r2/(1 - r1*r2) 
        return r1
        
class ModelInput:
    """ModelInput class - multi-layered photon scattering model input
        Class instance variables:
            Wth - play roulette if photon weight < Wth
            dz - z grid separation [cm]
            dr - r grid separation [cm]
            da - alpha grid separation [radian]
            nz - array range 0..nz-1
            nr - array range 0..nr-1
            na - array range 0..na-1
            layerObj - medium layer structure class instance
        Methods:
            
    """
    
    def __init__(self, nummua=0): 
        self.layerObj = LayerStruct(nummua)
        self.dz = modelfile[0].__getitem__(1)
        self.dr = modelfile[1].__getitem__(1)
        self.nz = int(modelfile[2].__getitem__(1))
        self.nr = int(modelfile[3].__getitem__(1))
        self.na = 10
        self.Wth = WEIGHT
        self.da = 0.5*np.pi/self.na

class MCMLModel(ModelInput):
    """MCMLModel class - multi-layered photon scattering model, inherits from
        ModelInput layer structure setup
        Class instance variables:
            Rsp - specular reflectance [-]
            Rd - total diffuse reflectance [-]
            A - total absorption probability [-]
            Tt - total transmittance [-]
            Rd_ra - 2D distribution of diffuse reflectance [1/(cm2 sr)]
            Rd_r - 1D radial distribution of diffuse reflectance [1/cm2]
            Rd_a - 1D angular distribution of diffuse reflectance [1/sr]
            A_rz - 2D probability density in turbid media over r & z [1/cm3]
            A_z - 1D probability density over z [1/cm]
            A_l - each layer's absorption probability [-]
            Tt_ra - 2D distribution of total transmittance [1/(cm2 sr)]
            Tt_r - 1D radial distribution of transmittance [1/cm2]
            Tt_a - 1D angular distribution of transmittance [1/sr]
        Methods:
            
    """

    def __init__(self, nummua=0):
        # extend the ModelInput base class instance variables
        ModelInput.__init__(self, nummua)
        self.numPhotons = 0
        # initialize the model grid arrays    
        self.Rsp = self.layerObj.calc_r_specular()
        self.Rd = 0.0
        self.A = 0.0
        self.Tt = 0.0
        self.Rd_ra = np.matrix(np.zeros((self.nr, self.na)))
        self.Rd_r = np.zeros(self.nr)
        self.Rd_a = np.zeros(self.na)
        self.A_rz = np.matrix(np.zeros((self.nr, self.nz)))
        self.A_z = np.zeros(self.nz)
        self.A_l = np.zeros(2 + self.layerObj.numLayers)
        self.Tt_ra = np.matrix(np.zeros((self.nr, self.na)))
        self.Tt_r = np.zeros(self.nr)
        self.Tt_a = np.zeros(self.na)

# 主要MCML光子行進的程序碼
def run_photon_simulation(model, wavelength, N):
    print('processing:'+str(wavelength)+'/total:316')
    ## --取出model參數start-- ##
    numloops = N
    numLayers = model.layerObj.numLayers
    list_ = LayerStruct(wavelength).layer
    tissuelayers = []
    for i in range(numLayers+2):
        tissuelayers.append([list_[i].n, list_[i].mua, list_[i].mus, list_[i].g])
    layerThickness = model.layerObj.layerThickness
    layerZ = model.layerObj.layerZ
    cosCrit = model.layerObj.cosCrit
    dz = model.dz
    dr = model.nz
    nz = model.nz
    nr = model.nr
    na = model.na
    Wth = model.Wth
    da = model.da
    Rsp = model.Rsp
    Rd = model.Rd
    Tt = model.Tt
    Rd_ra = model.Rd_ra
    Rd_r = model.Rd_r
    Rd_a = model.Rd_a
    Tt_ra = model.Tt_ra
    Tt_r = model.Tt_r
    Tt_a = model.Tt_a
    ## --取出model參數end-- ##

   ### --do_one_run-- #start
    for i in range(numloops):
        ## --光子旅程-- ##start
        # initial光子狀態
        numPhotons = 0
        x = 0.0
        y = 0.0
        z = 0.0
        ux = 0.0
        uy = 0.0
        uz = 1.0
        w = 1.0 - Rsp
        dead = False
        layer_index = 1
        s = 0
        sleft = 0
        # 光子開始活動
        # layer:[n,mua,mus,g]
        # 空氣層後的第一層(index=1)是否為air
        if (tissuelayers[layer_index][1] == 0.0) and (tissuelayers[layer_index][2] == 0.0):
            layer_index = 2     # 直接跳往下一層
            z = layerZ[layer_index][0]      # use z0 from the next layer
        
        ## --run_one_photon-- ##start
        while dead == False:   
            # hop_drop_spin

            # hop_in_glass
            if (tissuelayers[layer_index][1] == 0.0) and (tissuelayers[layer_index][2] == 0.0):
                
                if uz == 0.0:   # horizontal photon in glass is killed
                    dead = True
                else:           # Move the photon packet in glass layer.
                    # step_size_in_glass
                    if uz > 0.0:
                        dl_b = (layerZ[layer_index][1] - z)/uz
                    elif uz < 0.0:
                        dl_b = (layerZ[layer_index][0] - z)/uz
                    else:
                        dl_b = 0.0
                    s = dl_b
                    # hop
                    x += s*ux
                    y += s*uy
                    z += s*uz
                    # cross_or_not
                    if uz < 0.0:    # cross_up_or_not
                    
                        # 光子是否在目前層的上邊界（uz <0）穿透或反射。
                        # 如果“層”是第一層，若PARTIALREFLECTION設置為1，則光子將被部分穿透並部分反射
                        # 如果PARTIALREFLECTION設置為0，則光子將直接計算記錄往上穿透的光子權重作為反射率r。
                        # 如果“層”不是第一層並且光子往上穿透，將光子移動到“圖層-1”。
                        # 更新光子parmameters。    
                    
                        r = 0.0     # reflectance
                        ni = tissuelayers[layer_index][0]
                        nt = tissuelayers[layer_index-1][0]
                        if -uz <= cosCrit[layer_index][0]:
                            r = 1.0     # total internal reflection
                        else:
                            # --RFresnel(n1, n2, ca1)-- #
                            n1, n2, ca1 = ni, nt, -uz
                            if n1 == n2:			# matched boundary
                                ca2 = ca1
                                r = 0.0
                            elif ca1 > COSZERO:     # normal incident
                                ca2 = ca1
                                r = (n2-n1)/(n2+n1)
                                r *= r
                            elif ca1 < COS90D:      # very slant
                                ca2 = 0.0
                                r = 1.0
                            else:           # general	
                                # sine of the incident and transmission angles
                                sa1 = (1.0 - ca1*ca1)**0.5
                                sa2 = n1*sa1/n2
                                if sa2 >= 1.0:
                                    # double check for total internal reflection
                                    ca2 = 0.0
                                    r = 1.0
                                else:
                                    # cosines of the sum ap or
                                    # difference am of the two
                                    # angles. ap = a1+a2
                                    # am = a1 - a2     
                                    ca2 = (1.0 - sa2*sa2)**0.5;     
                                    cap = ca1*ca2 - sa1*sa2     # c+ = cc - ss
                                    cam = ca1*ca2 + sa1*sa2     # c- = cc + ss
                                    sap = sa1*ca2 + ca1*sa2     # s+ = sc + cs
                                    sam = sa1*ca2 - ca1*sa2     # s- = sc - cs
                                    r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam)
                            uz1 = ca2
                            # --RFresnel(n1, n2, ca1)-- #
                        if np.random.random_sample() > r:   # transmitted to layer-1
                            if layer_index == 1:
                                uz = -uz1
                                # --record_R-- #
                                ir = int((x*x + y*y)**0.5/dr)
                                if ir > (nr - 1):
                                    ir = (nr -1)
                                ia = int(np.arccos(-uz)/da) 
                                if ia > (na -1):
                                    ia = (na -1)
                                Rd_ra[ir, ia] += w*(1.0 - 0.0)
                                w *= 0.0
                                # --record_R-- #
                                dead = True
                            else:
                                layer_index -= 1
                                ux *= ni/nt
                                uy *= ni/nt
                                uz = -uz1
                        else:                               # reflected
                            uz = -uz
                        
                    else:           # cross_dn_or_not
                        r = 0.0
                        ni = tissuelayers[layer_index][0]
                        nt = tissuelayers[layer_index+1][0]
                        if uz <= cosCrit[layer_index][1]:
                            r = 1.0     # TIF
                        else:
                            # --RFresnel(n1, n2, ca1)-- #
                            n1, n2, ca1 = ni, nt, uz
                            if n1 == n2:			# matched boundary
                                ca2 = ca1
                                r = 0.0
                            elif ca1 > COSZERO:     # normal incident
                                ca2 = ca1
                                r = (n2-n1)/(n2+n1)
                                r *= r
                            elif ca1 < COS90D:      # very slant
                                ca2 = 0.0
                                r = 1.0
                            else:           # general	
                                # sine of the incident and transmission angles
                                sa1 = (1.0 - ca1*ca1)**0.5
                                sa2 = n1*sa1/n2
                                if sa2 >= 1.0:
                                    # double check for total internal reflection
                                    ca2 = 0.0
                                    r = 1.0
                                else:
                                    # cosines of the sum ap or
                                    # difference am of the two
                                    # angles. ap = a1+a2
                                    # am = a1 - a2     
                                    ca2 = (1.0 - sa2*sa2)**0.5;     
                                    cap = ca1*ca2 - sa1*sa2     # c+ = cc - ss
                                    cam = ca1*ca2 + sa1*sa2     # c- = cc + ss
                                    sap = sa1*ca2 + ca1*sa2     # s+ = sc + cs
                                    sam = sa1*ca2 - ca1*sa2     # s- = sc - cs
                                    r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam)
                            uz1 = ca2
                            # --RFresnel(n1, n2, ca1)-- #
                        if np.random.random_sample() > r:	# transmitted to layer+1
                            if layer_index == numLayers:
                                uz = uz1
                                # --record_T-- #
                                ir = int((x*x + y*y)**0.5/dr)
                                if ir > (nr - 1):
                                    ir = (nr - 1)
                        
                                ia = int(np.arccos(uz)/da)
                                if ia > (na - 1):
                                    ia = (na - 1)
                                Tt_ra[ir, ia] += w*(1.0 - 0.0)
                                w *= 0.0
                                # --record_T-- #
                                dead = True
                            else:
                                layer_index += 1
                                ux *= ni/nt
                                uy *= ni/nt
                                uz = uz1
                        else:                           # reflected
                            uz = -uz
            
            # hop_drop_spin_in_tissue
            else:
                # step_size_in_tissue
                mua = tissuelayers[layer_index][1]
                mus = tissuelayers[layer_index][2]
                if sleft == 0.0:    # make a new step
                    rnd = np.random.random_sample()
                    s = -np.log(rnd)/(mua + mus)
                else:               # take the leftover
                    s = sleft/(mua + mus)
                    sleft = 0.0
                # hit_boundary?
                if uz > 0.0:
                    dl_b = (layerZ[layer_index][1] - z)/uz
                elif uz < 0.0:
                    dl_b = (layerZ[layer_index][0] - z)/uz
                if (uz != 0.0) and (s > dl_b):
                    mut = tissuelayers[layer_index][1] + \
                        tissuelayers[layer_index][2]
                    sleft = (s - dl_b)*mut
                    s = dl_b
                    hit = True
                else:
                    hit = False

                if hit == True: # hit_boundary
                    # hop
                    x += s*ux
                    y += s*uy
                    z += s*uz
                    # cross_or_not
                    if uz < 0.0:    # cross_up_or_not
                    
                        # 光子是否在目前層的上邊界（uz <0）穿透或反射。
                        # 如果“層”是第一層，若PARTIALREFLECTION設置為1，則光子將被部分穿透並部分反射
                        # 如果PARTIALREFLECTION設置為0，則光子將直接計算記錄往上穿透的光子權重作為反射率r。
                        # 如果“層”不是第一層並且光子往上穿透，將光子移動到“圖層-1”。
                        # 更新光子parmameters。                    
                        r = 0.0     # reflectance
                        ni = tissuelayers[layer_index][0]
                        nt = tissuelayers[layer_index-1][0]
                        if -uz <= cosCrit[layer_index][0]:
                            r = 1.0     # total internal reflection
                        else:
                            # --RFresnel(n1, n2, ca1)-- #
                            n1, n2, ca1 = ni, nt, -uz
                            if n1 == n2:			# matched boundary
                                ca2 = ca1
                                r = 0.0
                            elif ca1 > COSZERO:     # normal incident
                                ca2 = ca1
                                r = (n2-n1)/(n2+n1)
                                r *= r
                            elif ca1 < COS90D:      # very slant
                                ca2 = 0.0
                                r = 1.0
                            else:           # general	
                                # sine of the incident and transmission angles
                                sa1 = (1.0 - ca1*ca1)**0.5
                                sa2 = n1*sa1/n2
                                if sa2 >= 1.0:
                                    # double check for total internal reflection
                                    ca2 = 0.0
                                    r = 1.0
                                else:
                                    # cosines of the sum ap or
                                    # difference am of the two
                                    # angles. ap = a1+a2
                                    # am = a1 - a2     
                                    ca2 = (1.0 - sa2*sa2)**0.5;     
                                    cap = ca1*ca2 - sa1*sa2     # c+ = cc - ss
                                    cam = ca1*ca2 + sa1*sa2     # c- = cc + ss
                                    sap = sa1*ca2 + ca1*sa2     # s+ = sc + cs
                                    sam = sa1*ca2 - ca1*sa2     # s- = sc - cs
                                    r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam)
                            uz1 = ca2
                            # --RFresnel(n1, n2, ca1)-- #
                        if np.random.random_sample() > r:   # transmitted to layer-1
                            if layer_index == 1:
                                uz = -uz1
                                # --record_R-- #
                                ir = int((x*x + y*y)**0.5/dr)
                                if ir > (nr - 1):
                                    ir = (nr -1)
                                ia = int(np.arccos(-uz)/da) 
                                if ia > (na -1):
                                    ia = (na -1)
                                Rd_ra[ir, ia] += w*(1.0 - 0.0)
                                w *= 0.0
                                # --record_R-- #
                                dead = True
                            else:
                                layer_index -= 1
                                ux *= ni/nt
                                uy *= ni/nt
                                uz = -uz1
                        else:                               # reflected
                            uz = -uz
                        
                    else:           # cross_dn_or_not
                        r = 0.0
                        ni = tissuelayers[layer_index][0]
                        nt = tissuelayers[layer_index+1][0]
                        if uz <= cosCrit[layer_index][1]:
                            r = 1.0     # TIF
                        else:
                            # --RFresnel(n1, n2, ca1)-- #
                            n1, n2, ca1 = ni, nt, uz
                            if n1 == n2:			# matched boundary
                                ca2 = ca1
                                r = 0.0
                            elif ca1 > COSZERO:     # normal incident
                                ca2 = ca1
                                r = (n2-n1)/(n2+n1)
                                r *= r
                            elif ca1 < COS90D:      # very slant
                                ca2 = 0.0
                                r = 1.0
                            else:           # general	
                                # sine of the incident and transmission angles
                                sa1 = (1.0 - ca1*ca1)**0.5
                                sa2 = n1*sa1/n2
                                if sa2 >= 1.0:
                                    # double check for total internal reflection
                                    ca2 = 0.0
                                    r = 1.0
                                else:
                                    # cosines of the sum ap or
                                    # difference am of the two
                                    # angles. ap = a1+a2
                                    # am = a1 - a2     
                                    ca2 = (1.0 - sa2*sa2)**0.5;     
                                    cap = ca1*ca2 - sa1*sa2     # c+ = cc - ss
                                    cam = ca1*ca2 + sa1*sa2     # c- = cc + ss
                                    sap = sa1*ca2 + ca1*sa2     # s+ = sc + cs
                                    sam = sa1*ca2 - ca1*sa2     # s- = sc - cs
                                    r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam)
                            uz1 = ca2
                            # --RFresnel(n1, n2, ca1)-- #
                        if np.random.random_sample() > r:	# transmitted to layer+1
                            if layer_index == numLayers:
                                uz = uz1
                                # --record_T-- #
                                ir = int((x*x + y*y)**0.5/dr)
                                if ir > (nr - 1):
                                    ir = (nr - 1)
                        
                                ia = int(np.arccos(uz)/da)
                                if ia > (na - 1):
                                    ia = (na - 1)
                                Tt_ra[ir, ia] += w*(1.0 - 0.0)
                                w *= 0.0
                                # --record_T-- #
                                dead = True
                            else:
                                layer_index += 1
                                ux *= ni/nt
                                uy *= ni/nt
                                uz = uz1
                        else:                           # reflected
                            uz = -uz
                else:           # same_layer
                    # hop
                    x += s*ux
                    y += s*uy
                    z += s*uz
                    # drop
                    iz = int(z/dz)
                    if iz > (nz - 1):
                        iz = (nz - 1)
                    ir = int((x*x + y*y)**0.5/dr)
                    if ir > (nr - 1):
                        ir = (nr - 1)
                    dwa = w * mua/(mua+mus)
                    w -=dwa
                    # spin
                    # --SpinTheta-- #
                    g = tissuelayers[layer_index][3]
                    if g == 0.0:
                        cost = 2*np.random.random_sample() - 1
                    else:
                        temp = (1 - g*g)/(1 - g + 2*g*np.random.random_sample())
                        cost = (1 + g*g - temp*temp)/(2*g)
                        if cost < -1:
                            cost = -1.0
                        elif cost > 1:
                            cost = 1.0
                    # --SpinTheta-- #
                    sint = (1.0 - cost*cost)**0.5
                    psi = 2.0*np.pi*np.random.random_sample()
                    cosp = np.cos(psi)
                    if psi < np.pi:
                        sinp = (1.0 - cosp*cosp)**0.5
                        # sqrt() is faster than sin().
                    else:
                        sinp = -(1.0 - cosp*cosp)**0.5
                    if np.fabs(uz) > COSZERO:   # normal incident
                        ux = sint*cosp
                        uy = sint*sinp
                        uz = cost*np.sign(uz)
                    else:                       # regular incident
                        temp = (1.0 - uz*uz)**0.5
                        ux = sint*(ux*uz*cosp - uy*sinp)/temp + ux*cost
                        uy = sint*(uy*uz*cosp + ux*sinp)/temp + uy*cost
                        uz = -sint*cosp*temp + uz*cost
            
            # roulette
            if (w < Wth) and (dead == False):
                if w == 0.0:
                    dead = True
                elif np.random.random_sample() < CHANCE:    # survived the roulette
                    w /= CHANCE
                else:
                    dead = True
        
        numPhotons += 1
        ## --run_one_photon-- ##end                     
        ## --光子旅程-- ##end
   ### --do_one_run-- #end

   ### --sum_scale_result-- #start
    for ir in range(nr):
        sum = 0.0
        for ia in range(na):
            sum += Rd_ra[ir, ia]
        Rd_r[ir] = sum
    for ia in range(na):
        sum = 0.0
        for ir in range(nr):
            sum += Rd_ra[ir, ia]
        Rd_a[ia] = sum
    sum = 0.0
    for ir in range(nr):
        sum += Rd_r[ir]
    Rd = sum

    scale1 = 4.0*np.pi*np.pi*dr*np.sin(da/2)*dr*numloops
    for ir in range(nr):  
            for ia in range(na):
                scale2 = 1.0/((ir+0.5)*np.sin(2.0*(ia+0.5)*da)*scale1)
                Rd_ra[ir, ia] *= scale2
    scale1 = 2.0*np.pi*dr*dr*numloops
    for ir in range(nr):
            scale2 = 1.0/((ir+0.5)*scale1)
            Rd_r[ir] *= scale2
    scale1  = 2.0*np.pi*da*numloops
    for ia in range(na):
            scale2 = 1.0/(np.sin((ia+0.5)*da)*scale1)
            Rd_a[ia] *= scale2    
    scale2 = 1.0/numloops
    Rd *= scale2
   ### --sum_scale_result-- #end
    return Rd

# 支持CPU多核心加速運算程式碼
def job(q, model, N, boundary):
    R_df = []
    for wavelength in range(boundary[0], boundary[1], 5):
        Rd = run_photon_simulation(model, wavelength, N)
        R_df.append(Rd)
    q.put(R_df)

if __name__ == "__main__":

    save = input('save the Reflection Spectrum?(y/n):')
    if save == 'y':
        print ('save ON !')
        filename = input('the filename?(do NOT add extension):')
    else:
        print ('save OFF !')
    n = input('How many photons for simulation?(1000 photons spend about 4 mins):')
    N = int(n)
    cpu_number = input('How many multicore operations to use?')
    cpu_number = int(cpu_number)
    

    # 本程式只有製作到8核運算，如果想擴充更多核心運算，請修改以下cpu_number<=8 的8改成想要的最高核心數
    if cpu_number>=1 and cpu_number<=8 and cpu_number<=cpu_count(): 
        cpu_number = int(cpu_number)
    elif cpu_number>cpu_count():
        cpu_number = cpu_count()
    else:
        cpu_number = 1

    tStart = time.time()
    model = MCMLModel()
    c = {}

    # 本程式只有製作到8核運算，如果想擴充更多核心運算，請增加以下程式碼如:
    # 在m8.start()以下依序增加核心數 
    #   if c.get(8):
    #     boundary = c[8]
    #     q9 = Queue()
    #     m9 = mp.Process(target=job, args=(q9,model,N,boundary))
    #     m9.start()
    # 以此類推..... 
    for i in range(0,cpu_number):
        c[i] = [int(0+i*316/cpu_number), int((i+1)*316/cpu_number)]
    if c.get(0):
        boundary = c[0]
        q1 = Queue()
        m1 = mp.Process(target=job, args=(q1,model,N,boundary))
        m1.start()
    if c.get(1):
        boundary = c[1]
        q2 = Queue()
        m2 = mp.Process(target=job, args=(q2,model,N,boundary))
        m2.start()
    if c.get(2):
        boundary = c[2]
        q3 = Queue()
        m3 = mp.Process(target=job, args=(q3,model,N,boundary))
        m3.start()
    if c.get(3):
        boundary = c[3]
        q4 = Queue()
        m4 = mp.Process(target=job, args=(q4,model,N,boundary))
        m4.start()
    if c.get(4):
        boundary = c[4]
        q5 = Queue()
        m5 = mp.Process(target=job, args=(q5,model,N,boundary))
        m5.start()
    if c.get(5):
        boundary = c[5]
        q6 = Queue()
        m6 = mp.Process(target=job, args=(q6,model,N,boundary))
        m6.start()
    if c.get(6):
        boundary = c[6]
        q7 = Queue()
        m7 = mp.Process(target=job, args=(q7,model,N,boundary))
        m7.start()
    if c.get(7):
        boundary = c[7]
        q8 = Queue()
        m8 = mp.Process(target=job, args=(q8,model,N,boundary))
        m8.start()                        
    print('Currently using %d cores in the operation'%(i+1))

    # 本程式只有製作到8核運算，如果想擴充更多核心運算，請增加以下程式碼如:
    # 在R.append(q8.get())以下依序增加核心數 
    #  if c.get(8):
    #      m9.join()
    #      R.append(q9.get())
    # 以此類推..... 
    R = []
    if c.get(0):
        m1.join()
        R.append(q1.get())
    if c.get(1):
        m2.join()
        R.append(q2.get())
    if c.get(2):
        m3.join()
        R.append(q3.get())
    if c.get(3):
        m4.join()
        R.append(q4.get())
    if c.get(4):
        m5.join()
        R.append(q5.get())
    if c.get(5):
        m6.join()
        R.append(q6.get())
    if c.get(6):
        m7.join()
        R.append(q7.get())
    if c.get(7):
        m8.join()
        R.append(q8.get())
    print('operation done!')

    import itertools
    out = list(itertools.chain.from_iterable(R))
    tEnd = time.time()#計時結束
    print ('spend time:'+str(tEnd - tStart)+'seconds')
    if save == 'y':
        np.savetxt(str(filename)+'.csv', out, delimiter='\t')
    path = os.path.dirname(__file__)
    WL = np.loadtxt(os.path.join(path,'wavelength.csv'))
    plt.plot(WL[::5],out)
    plt.show()
    print ('finish!')
    input('Press any key to end')
