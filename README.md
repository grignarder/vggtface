# VGGTFace

**Paper:** *VGGTFace: Topologically Consistent Facial Geometry Reconstruction in the Wild* (AAAI26)

VGGTFace is a system for reconstructing a topologically consistent 3D facial mesh from multi-view face images captured in the wild in less than 10 seconds.

⭐ Our new work **VGGTFace2** has been accepted to SIGGRAPH Asia 2026! VGGTFace2 achieves faster inference speed and significantly higher reconstruction quality than VGGTFace. We will release the source code of VGGTFace2 soon – stay tuned!

## 🔗 Online Demo

Try VGGTFace in your browser (upload 16 images → reconstruct a mesh in one click)!

- **Gradio Demo:** https://uu31763-9195-9f4b0121.westc.seetacloud.com:8443/

> **Note:** The current web demo supports **exactly 16 images** per reconstruction.
>
> ⭐ If you find VGGTFace useful, please consider giving this repository a star!

## 🚀 Getting Started

### 1. Install Pixel3DMM

First, install **Pixel3DMM** following the instructions in their official repository:
https://github.com/SimonGiebenhain/pixel3dmm.

We rely on its pretrained UV predictor weights, but we do not require FLAME-related assets.
Therefore, when running `install_preprocessing_pipeline.sh`, you only need to download the UV predictor weights.
Running `download_flame2023.sh` is **not necessary** for VGGTFace.

### 2. Download VGGT pretrained weights

Please download the pretrained weights from the official VGGT repository:

https://github.com/facebookresearch/vggt

After downloading, rename the checkpoint to `vggt_weights.pt` and place it under:

`./pretrained_weights/vggt_weights.pt`

> Note: Please follow VGGT’s official instructions and license terms when downloading and using the weights.

### 3. Preprocess multi-view images

To preprocess multi-view images, run:

`python preprocess.py --image_folder {image_folder} --output_folder {output_folder}`

During preprocessing, we use Pixel3DMM’s UV predictor to estimate a UV map for each image,
and use facer to estimate the mask for each image.

Example:

`python preprocess.py --image_folder ./examples/example1 --output_folder ./preprocessed_data/example1`

Note: 

1. You may need to install **pyfacer**—just follow the instructions in their official repository: [https://github.com/FacePerceiver/facer](https://github.com/FacePerceiver/facer).

2. You may run into the error `forward() got an unexpected keyword argument 'bbox_scale_factor'`. If so, simply remove the `bbox_scale_factor=1.25` argument in `preprocess.py`. This seems to be caused by API inconsistencies across different versions of pyfacer.

### 4. Reconstruct facial mesh

After preprocessing, run the reconstruction script:

`python vggtface_infer.py --BASE_PATHS {preprocessed_dir}`

Example:

`python vggtface_infer.py --BASE_PATHS ./preprocessed_data/example1`

Once finished, you will find `result.ply` under the corresponding directory, which is the reconstructed mesh.

#### Batch reconstruction (recommended)

You can also reconstruct multiple multi-view sets sequentially in a single run (so VGGT weights are loaded only once). Provide multiple directories separated by commas:

`python vggtface_infer.py --BASE_PATHS ./preprocessed_data/example1,./preprocessed_data/example2`

## 📝 Paper

- Title: *VGGTFace: Topologically Consistent Facial Geometry Reconstruction in the Wild*  
- PDF / arXiv: https://arxiv.org/abs/2511.20366

If you use this work, please consider citing our paper. 

```
@inproceedings{ming2026vggtface,
  title={VGGTFace: Topologically Consistent Facial Geometry Reconstruction in the Wild},
  author={Ming, Xin and Han, Yuxuan and Huang, Tianyu and Xu, Feng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={10},
  pages={8080--8088},
  year={2026}
}
```

## 👥 Authors & Affiliation

**Authors:** Xin Ming, Yuxuan Han, Tianyu Huang, Feng Xu
**Affiliation:** BNRist and School of Software, Tsinghua University

## 📌 Notes

- The demo currently reloads some components per run, so inference may be slow. We are working on optimizing it.

## 📄 License

This project is released under the MIT License (see `LICENSE`).
It depends on third-party projects (e.g., Pixel3DMM and VGGT) that are distributed under their own licenses.
Please make sure you comply with the corresponding upstream terms when using their code/models.

## 📬 Contact

Questions, feedback, or collaboration ideas are welcome!

- Email: 1729406968@qq.com
- GitHub Issues: please open an issue on this repository.
