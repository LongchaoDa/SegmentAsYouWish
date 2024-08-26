# To get started with this repo

pip install omegaconf
sudo apt-get install gfortran
pip install escnn
pip install kornia


Get the example training data ready: 

cd SegmentAsYouWish
mkdir data

[link] https://zenodo.org/records/7860267

cd data
wget -O FLARE22Train.zip "https://zenodo.org/records/7860267/files/FLARE22Train.zip?download=1"
unzip FLARE22Train.zip
rm -rf FLARE22Train.zip

Now you should have the file folder of "images" and "labels" at: /home/local/ASURITE/longchao/Desktop/project/GE_health/SegmentAsYouWish/data/FLARE22Train

Then you need to run `python pre_CT_MR.py`, in order to run it, you need to install the following packages: 

pip install SimpleITK
pip install scikit-image
pip install connected-components-3d

