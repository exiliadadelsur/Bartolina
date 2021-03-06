# Bartolina
![Alt text](/logo/Bartolina.png?raw=true)

[![Bartolina CI](https://github.com/exiliadadelsur/Bartolina/actions/workflows/bartolina_ci.yml/badge.svg)](https://github.com/exiliadadelsur/Bartolina/actions/workflows/bartolina_ci.yml)

Bartolina is a real space reconstruction algorithm that corrects galaxy positions by Kaiser and Finger of God (FoG) effects. 
We follow the work carried out in [Shi et al. 2016, ApJ, 833, 241](https://iopscience.iop.org/article/10.3847/1538-4357/833/2/241/pdf) and [Wang et al. 2012, MNRAS, 420, 1809](https://arxiv.org/pdf/1108.1008.pdf).

Bartolina has the principal class:
* **ReZspace**: Sets parameters such as the cosmology to use and allows a set of methods. 

Bartolina has the following methods:
* **dark_matter_halos**: Creates Halo and GalInGroup objects.
* **xyzcoordinates**: Obtains cartesian coordinates of halos centers.
* **groups**: Finds groups of galaxies.
* **radius**: Obtains radius of dark matter halos
* **centers**: Finds halos centers.
* **group_prop**: Obtaines properties of halos.
* **bias**: Calculate halo bias function.
* **dc_fog_corr**: Corrects comoving distance only considering FoG effect.
* **z_fog_corr**: Corrects redshift only considering FoG effect.
* **grid3d**: Create a three dimensional grid.
* **grid3d_axislim**: Determine the minimum and maximum xyz coordinates.
* **grid3d_gridlim**: Determine the limits of the grid.
* **grid3dcells**: Division of the box in cells.
* **density**: Calculate the mass density in each cell.
* **calcf**: Compute the approximation to the function f (omega).
* **zkaisercorr**: Corrects redshift only considering Kaiser effect.
* **kaisercorr**: Corrects the Kaiser effect only.
* **fogcorr**: Corrects the Finger of God effect only.
* **realspace**: Corrects both effects (Kaiser and FoG).

## Requirements

You need Python 3.8 to run Bartolina

## Development Install

Clone this repo and then inside the local directory execute

```
$ pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE) file for details.

## About Bartolina Sisa

Bartolina Sisa (25 August 1750 - 5 September 1782) is an Aymara national heroine. 

Through her trade as a merchant, she learned about the subjugation, exploitation, offenses and abuse suffered by the Andean peoples by the Spanish. That is why she assumed the conviction to redeem her people from oppression and to fight for definitive emancipation. 

She joined in the struggles with her husband Tupac Katari to organize and lead different uprisings. When the Aymara-Quechua indigenous insurgency broke out in 1781, she proclaimed herself Virreina Inca. Her army had around 80,000 combatants. 

It was betrayed and handed over to the Spanish. She was killed by being dragged by a horse to death. Then she was dismembered and her body parts were exhibited in different places of the ayllus where she resisted in fight.

Bartolina is one of the most emblematic symbols of the anticolonial struggles of the 18th century in Latin America.
Every September 5, the International Day of Indigenous Women is celebrated in her honor.

Bartolina Sisa is perhaps the most famous name among women who fought in indigenous uprisings, but not the only one. Other women were Tomasa Titu Condemayta, Micaela Bastidas Puyucagua, Manuela Condori, Gregoria Apaza and many other renowned heroines and many others anonymous.

## Authors

Noelia Rocío Perez and Claudio Antonio Lopez Cortez



