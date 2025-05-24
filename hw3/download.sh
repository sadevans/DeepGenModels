git clone https://github.com/NVlabs/stylegan2-ada-pytorch
cd stylegan2-ada-pytorch
git clone https://github.com/denkogit/stylegan2_models

mkdir pretrained_models
cd pretrained_models
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1cUv_reLE6k3604or78EranS7XzuVMWeO" -O "e4e_ffhq_encode.pt"


bzip2 -d /netapp/a.gorokhova/itmo/DeepGenModels/hw3/stylegan2-ada-pytorch/pretrained_models/shape_predictor_68_face_landmarks.dat.bz2
pwd
cd ..
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1QIPdkYyIwqEUS8jBNouWo9eYzr3DwDd5" -O "ms1mv3_arcface_r50_fp16.pth"
cd ..

# cp ./e4e_ffhq_encode.pt ./stylegan2-ada-pytorch/pretrained_models/
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=10zYE9lkYq6SuMVL0MaTv4YSuwdg_Wbhg" -O "./editing.zip"

unzip ./editing.zip -d ./
rm -rf ./__MACOSX