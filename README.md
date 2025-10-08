If you want to run this yourself, you have to pull the celeb-a HQ dataset and then update the path in the prepare_dataset.py file, then run celebhqdiffusion_fancy.py (took about 12 hours to hit epoch 1000 on racecar), then update the Diffusion_Results.py to load whatever epoch you decided to stop at.

For those of you who have a racecar account, this is all in ```/data5/accounts/marsh/Diffusion/```, and is all pre-trained and ready to go

The diffusion model isn't optimal architecture, just the first one I found that worked sufficiently.

Pre-trained model weights are here: https://drive.google.com/file/d/1NV68JKtXhyE3wqCP72dgj3S3AyTZ_0dz/view?usp=sharing

Diffusion_Results_New.py is the current one
