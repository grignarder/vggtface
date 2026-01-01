# VGGTFace

**Paper:** *VGGTFace: Topologically Consistent Facial Geometry Reconstruction in the Wild* (AAAI26)

VGGTFace is a system for reconstructing a topologically consistent 3D facial mesh from multi-view face images captured in the wild in less than 10 seconds.

> 🚧 **Code Release:** We are preparing the open-source release.  
> In the meantime, you can try the public demo below.

## 🔗 Online Demo

Try VGGTFace in your browser (upload 16 images → reconstruct a mesh in one click)!

- **Gradio Demo:** https://uu31763-ba3f-b060fff2.westc.gpuhub.com:8443/

> **Note:** The current web demo supports **exactly 16 images** per reconstruction.

## 🚀 Getting Started

### 1. Install Pixel3DMM

First, install **Pixel3DMM** following the instructions in their official repository:
https://github.com/SimonGiebenhain/pixel3dmm.

We rely on its pretrained UV predictor weights, but we do not require FLAME-related assets.
Therefore, when running `install_preprocessing_pipeline.sh`, you only need to download the UV predictor weights.
Running `download_flame2023.sh` is **not necessary** for VGGTFace.

### 2. Preprocess multi-view images

To preprocess multi-view images, run:

`python preprocess.py --image_folder {image_folder} --output_folder {output_folder}`

During preprocessing, we use Pixel3DMM’s UV predictor to estimate a UV map for each image,
and use facer to estimate the mask for each image.

Example:

`python preprocess.py --image_folder ./examples/example1 --output_folder ./preprocessed_data/example1`

## 📝 Paper

- Title: *VGGTFace: Topologically Consistent Facial Geometry Reconstruction in the Wild*  
- PDF / arXiv: https://arxiv.org/abs/2511.20366

If you use this work, please consider citing our paper.

## 👥 Authors & Affiliation

**Authors:** Xin Ming, Yuxuan Han, Tianyu Huang, Feng Xu
**Affiliation:** BNRist and School of Software, Tsinghua University

## 📌 Notes

- The demo currently reloads some components per run, so inference may be slow. We are working on optimizing it.

## 📬 Contact

Questions, feedback, or collaboration ideas are welcome!

- Email: 1729406968@qq.com
- GitHub Issues: please open an issue on this repository.