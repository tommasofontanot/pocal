# POCAL

POCAL (Python Optical Coating Analysis Library) allows to easily design optical coatings both as single layer and as multilayer stack, monitoring a wide variety of optical properties and, moreover, giving the possibility to automatically refine the multilayer design to achieve the desired optical goals. The library is completely open source, and it can be easily coupled to several Python-based ray tracers or libraries to work on more complex simulations. The results have been extensively tested and are comparable with the ones obtained using commercial software.

## **Downloading POCAL**

Downloading the library is very easy using pip :

```
pip install pocal
```

Supporting files can be downloaded from: https://github.com/tommasofontanot/pocal/tree/main/resources

The folder with the resources needs to be made the current working directory.

```
import os
os.chdir(resources folder)
```

## **Cite POCAL**

If you use the library, please cite the paper published in Optics Continuum: [Fontanot, T., Bhaumik, U., Kishore, R. and Meuret, Y., 2023. POCAL: a Python-based library to perform optical coating analysis and design. Optics Continuum, 2(4), pp.810-824. DOI: https://doi.org/10.1364/OPTCON.484972](https://opg.optica.org/optcon/fulltext.cfm?uri=optcon-2-4-810&id=528721)

## **Using the library**

The library can be called as follows:

```
import pocal
import os
os.chdir('resources folder')
#upload the prescription file(here 3layers.txt), angle of incidence, minimum wavelength, maximum wavelength, wavelength resolution, 
#reference wavelength, optical thickness and type of refinement
test = pocal.pocal(r'3layers.txt',40,200,1500,1,600,False,None)
```


## **Defining the Prescription file**

The prescription file represents the stacking of the materials. An example is shown below:

+----Immersing Medium----+-Glass----------+

+----Layer3----------------+-TiO2-------------50 nm+

+----Layer2----------------+-SiO2-------------60 nm+

+----Layer1----------------+-TiO2-------------70 nm+

+----Substrate-------------+-Glass------------+


To simulate the proposed design, the prescription text file must be written as follows:

![image](https://user-images.githubusercontent.com/18205576/229857312-8f7d8ce8-ad8e-46c9-bd0b-ce8e2bbe67c4.png)


## **Python tutorial**

An in-depth tutorial to use the library can be found here: [POCAL Tutorial](https://github.com/tommasofontanot/pocal/blob/main/Pocal.ipynb)

## **License**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/tommasofontanot/pocal/blob/main/LICENSE.txt)
