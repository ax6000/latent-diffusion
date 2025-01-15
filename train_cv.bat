set sd_sscmd_continue = on
echo fold 0
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=0 data.params.validation.params.fold=0 data.params.test.params.fold=0
echo fold 1
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=1 data.params.validation.params.fold=1 data.params.test.params.fold=1
echo fold 2
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=2 data.params.validation.params.fold=2 data.params.test.params.fold=2
echo fold 3
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=3 data.params.validation.params.fold=3 data.params.test.params.fold=3
echo fold 4
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=4 data.params.validation.params.fold=4 data.params.test.params.fold=4
echo fold 5
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=5 data.params.validation.params.fold=5 data.params.test.params.fold=5
echo fold 6
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=6 data.params.validation.params.fold=6 data.params.test.params.fold=6
echo fold 7
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=7 data.params.validation.params.fold=7 data.params.test.params.fold=7
echo fold 8
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=8 data.params.validation.params.fold=8 data.params.test.params.fold=8
echo fold 9
python main.py --base .\configs\latent-diffusion\v4_ppg2abp-ldm-kl-4-concat-cv.yaml -t --scale_lr False data.params.train.params.fold=9 data.params.validation.params.fold=9 data.params.test.params.fold=9