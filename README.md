# MCML_simulate-spectroscopy
This code was created for simulating Diffuse Reflectance Spectroscopy(DRS) of human skin tissue via Monte Carlo for Multi-Layered media(MCML)

You may need to refer to this paper to understand the principle.

MCML: https://omlc.org/software/mc/mcml/index.html and https://omlc.org/software/mc/mcml/MCman.pdf

# Introduction
The technical details of the Monte-Carlo multilayer (MCML) simulation have been described in the literature [Meglinski and Matcher (2002)]. We followed the algorithm but rewrote the code in Python to significantly accelerate the simulation speed (via CPU multicore processing). </br>
In all simulations, 10<sup>4</sup> photon packets at each wavelength were used to enter the model from above (z<0) at the air at stratum corneum boundary. At this point a set of probabilities determine if the photon packets are reflected or enter the tissue. When a packet has entered the tissue, it can be partially or fully absorbed by event bins uniformly distributed throughout the tissue.With a starting photon weight W of 1, every time a photon packet interacts with a bin it loses part of its weight and then gets scattered in a direction determined by the anisotropy factor and scattering coefficient. At the end of the simulation, all diffusely scattered photons locating at the incident side (z<0) were added up to give a diffuse reflectance spectrum. </br>
One of the simulation results is presented in **below figure** with filled circles enclosed in error bars estimated from 10 simulations.
For comparison, a diffuse reflectance spectrum measured on the inside ofthe right forearm of an adult with our DRS apparatus is also presented (red solid curve). A good agreement with the simulation profile was obtained, suggesting that the model used captured the essential elements of human skin tissue.
![Imgur](https://i.imgur.com/cHXjQje.jpg "Monte-Carlo multilayer simulation (solid circles with error bars) and measured (red solid curve) diffuse reflectance spectrum of human skin tissue.")

# Usage
These file must be in same path with **run.py**</br>
+ wavelength.csv: all wavelength data for this program
+ model_input.txt: some parameters of the simulated tissue model
+ mua_water.csv: the optical absorption properties (μ<sub>a</sub> cm<sup>-1</sup>) of water
+ mua_oxy.csv: the optical absorption properties (μ<sub>a</sub> cm<sup>-1</sup>) of oxygenated whole blood
+ mua_deoxy.csv: the optical absorption properties (μ<sub>a</sub> cm<sup>-1</sup>) of deoxygenated whole blood
+ mua_melanin.csv: the optical absorption properties (μ<sub>a</sub> cm<sup>-1</sup>) ofinterior of typical cutaneous melanosome</br>

And these optical absorption properties are referred to https://omlc.org/software/mc/mcxyz/index.html
- - -
This code is a preset 8-multicore processing. If you have enough CPU cores, you must increase/revise the code.</br>
You must revise [line731](https://github.com/GarrettTW/MCML_simulate-spectroscopy/blob/33d8c457c14ce4164e525d4fda282cfcbaf2abc0/run.py#L731) the **8** to **number of your CPU cores**

```python
if cpu_number>=1 and cpu_number<=8 and cpu_number<=cpu_count(): 
```

You must increase the following code below [line791](https://github.com/GarrettTW/MCML_simulate-spectroscopy/blob/33d8c457c14ce4164e525d4fda282cfcbaf2abc0/run.py#L791) (if you want to use 9 cores for mutiprocessing)

```python
if c.get(8):
    boundary = c[8]
    q9 = Queue()
    m9 = mp.Process(target=job, args=(q9,model,N,boundary))
    m9.start()                        
```
and increase the following code below [line824](https://github.com/GarrettTW/MCML_simulate-spectroscopy/blob/33d8c457c14ce4164e525d4fda282cfcbaf2abc0/run.py#L824)

```python
if c.get(8):
    m9.join()
    R.append(q9.get())                     
```
- - -
Run the `run.py`

Enter your requirements as prompted

`save the Reflection Spectrum?(y/n):`You can type`y`or`n`to choose whether to store the simulated spectrum on your hard drive.
    
`the filename?(do NOT add extension):`If you type`y`, you need to type the file name (you do **not** need to enter the **filename extension**)
    
`How many photons for simulation?(1000 photons spend about 4 mins)`more photons will result in better spectrum (but you will spend more time)

`How many multicore operations to use?`Please consider the number of your CPU cores
 
