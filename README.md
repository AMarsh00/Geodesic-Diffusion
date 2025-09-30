Here is the current code I have and the pdf with my math in it -- poorly implemented geodesics, just a proof of concept.

If you want to run this yourself, you have to pull the celeb-a HQ dataset and then update the path in the prepare_dataset.py file, then run celebhqdiffusion_fancy.py (took about 12 hours to hit epoch 1000 on racecar), then update the Diffusion_Results.py to load whatever epoch you decided to stop at.

For those of you who have a racecar account, this is all in ```/data5/accounts/marsh/Diffusion/```, and is all pre-trained and ready to go.

I wasn't able to upload the t=500 and t=999 animations due to size restrictions, but here they are in imgur: https://imgur.com/a/UryNNNi

Pre-trained model weights are in the vp_diffusion_outputs folder
