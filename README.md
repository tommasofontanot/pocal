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

+----Layer3----------------+-TiO2-------------50 nm+

+----Layer2----------------+-SiO2-------------60 nm+

+----Layer1----------------+-TiO2-------------70 nm+

+----Substrate-------------+-Glass------------+


To simulate the proposed design, the prescription text file must be written as follows:

![image](https://user-images.githubusercontent.com/18205576/229857312-8f7d8ce8-ad8e-46c9-bd0b-ce8e2bbe67c4.png)


## **Python tutorial**

An in-depth tutorial to use the library can be found here: [https://github.com/tommasofontanot/pocal/blob/main/Pocal.ipynb](POCAL)
