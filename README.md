# Bartolina

Bartolina is a real space reconstruction algorithm that corrects galaxy positions by Kaiser and Finger of God (FoG) effects. 
We follow the work carried out in [Shi et al. 2016, ApJ, 833, 241](https://iopscience.iop.org/article/10.3847/1538-4357/833/2/241/pdf) and [Wang et al. 2012, MNRAS, 420, 1809](https://watermark.silverchair.com/mnras0420-1809.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAArgwggK0BgkqhkiG9w0BBwagggKlMIICoQIBADCCApoGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMEhE4wSh1Y4O5XeUWAgEQgIICazOIdRFy7IysqA9fMx4B92bTxAws4pzEtxim1UbeBaZ64iQy_ytVC9HY4DsKIHq4wM09gW84mI-uggU2KHN7BAoi9Ps52HYw_QA-C2Vi_MClrMi2zQBVi18Tmdv5kEdDE8iBJDqMjm_Isw0nv9kk41RGnaC_TvnNQ_ugEGlkiMJaA6xwT2jIniikSBl1xnHAm-6WFTDMrqDqKEknuGz6L4ngFuHiJ1lv-lfy2dMJBjwYam9BtAg6bRM3qY59nIJgUBLM-j66A2UuS1zZIrWcaa04IF7GSn2Zh1lqF0K75v5kkGIEDzJDzl2SBr3-tIRnFxb7N2qvM2Y8V_Kxb_LPUme0dtp485AISJ1pmZIL8YbdFgG7yuai3WxsNDOWEO0r7w4OWqOkSUw58NKKf3069MiKpFvIHAej3hJhxDdJbyrnQLoUZybHnpgfLOR2i5OrKMqLJn6mWFVkt9O7pP--QJ0BwIZDTcaceVL3eO1ndhvj4EMAwBrmBSMW3WfNbFIeZDzlCh1v_6T7r6MfgLcuSALZLPFh6iHT9_TWPApqYxUH21p1kdwIrDLYqHoJ_Bz5--5JUhza7H6jkvr3Gq9M3ZwoIHrnU5K_aiUo1bka93_8c-Qv0yYN0dFLqqPWrh_lo28GiC0kvBDcW168wIMux9o18ebmIh4etcAE2Jn9cyPPZDIhlrkgqGEY8xlf4ICgdpNbhwwGMGLxNDWZoNqzXKPqknl9TUq8xezB3dq5cTeUr4YoV68zKwokV62O4qvawNGSpbT4BZC06xgiJ_7vXiWKN7zMmczwimhdBkX4Y1R0nDpiRlIzzZj5pNQ).

Bartolina has the principal class:
* **ReZspace**: Sets parameters such as the cosmology to use and allows a set of methods. 

Bartolina has the following methods:
* **Halos**: Find massive dark matter haloes and cartesian coordinates of his centers. Necesary for all the other methods.
* **Kaisercorr**: Corrects the Kaiser effect only.
* **FoGcorr**: Corrects the Finger of God effect only.
* **RealSpace**: Corrects both effects (Kaiser and FoG).

## Requirements

You need Python 3.8 to run Bartolina

## Development Install

Clone this repo and then inside the local directory execute

```
$ pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE) file for details.

## Authors

Noelia Rocío Perez and Claudio Antonio Lopez Cortez





