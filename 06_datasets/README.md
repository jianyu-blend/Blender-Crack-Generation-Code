# Datasets

This directory holds no data. The generated images are released separately, so that the code
and the data can be downloaded independently.

| Dataset | Where |
|---|---|
| BCG synthetic images and labels | `<dataset repository URL — fill in on release>` |
| MCrack1300 real images and annotations | [MCrack1300 on Roboflow](https://universe.roboflow.com/acsalab/masonry-zqhaw) |

The dataset repository carries the full description: the directory layout, the class order, what
the filenames encode, the checksums and a verification script. Once the dataset is extracted,
point the downstream configuration at it:

```yaml
downstream:
  synthetic_pool: /path/to/BCG
```

The real images come from MCrack1300 and are not redistributed by this repository. Download a
YOLO-segmentation export from the link above and point `dataset_root` at the extracted folder.

> Ye, Z., Lovell, L., Faramarzi, A. and Ninic, J. (2024). SAM-based instance segmentation
> models for the automation of structural damage detection. *Advanced Engineering Informatics*
> 62, 102826. [doi:10.1016/j.aei.2024.102826](https://doi.org/10.1016/j.aei.2024.102826) ·
> [arXiv:2401.15266](https://arxiv.org/abs/2401.15266)

Request it from its authors or obtain it through the publication above. See the repository
`README.md` for how its partitions are used.
