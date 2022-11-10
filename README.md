# POCAL

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

+----Layer3----------------+-TiO2-------------+

+----Layer2----------------+-SiO2-------------+

+----Layer1----------------+-TiO2-------------+

+----Substrate-------------+-Glass------------+
