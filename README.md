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
test = pocal.pocal(r'3layers.txt',40,200,1500,1,600,False,None)
```


