# Aria Glasses Utils

**Utilities to extract, manage and stream live or recorded data from Meta [Project Aria](https://www.projectaria.com/) glasses.**

> ⚠️ The repository is still working progress and it is not made to be used in a production environment. Please instead refer to the official documentation of the project.  

---

![Glasses structure](https://facebookresearch.github.io/projectaria_tools/assets/images/aria_hardware_diagram-e9a6473cfce8c11ef316c01ade49fe09.png)

The package has several script that can be uses both as singleton or as a package. For instance, to trigger the object detection script:

```sh
python src/objectDetection.py --interface usb --update_iptables
```


