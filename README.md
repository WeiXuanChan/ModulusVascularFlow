# ModulusVascularFlow
Multicase vascular flow PINN based on Nvidia Modulus v20.09 framework

NVIDIA Modulus:https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/index.html#undefined

Codes extention from NVIDIA Modulus can be found in folder ModulusDL

## Examples
#### Multicase PINN for 2D stenosis
stenosis2dsimplemode_plus_0hb.py : plus size Modes Network (codes fully commented)
	
stenosis2dsimplemode_0hb.py : full size Modes Network
	
stenosis2dsimplemode_plus_deqn_0hb.py : plus size Modes Network with added derivatives of governing and boundary equations with respect to case parameter

stenosis2dsimplecase_0hb.py : full size Hypernetwork

stenosis2dsimplecase_low_0hb.py : small size Hypernetwork
	
stenosis2dsimplecase_plus_0hb.py : plus size Hypernetwork
	
stenosis2dsimplecase_plus_deqn_0hb.py : plus size Hypernetwork with added derivatives of governing and boundary equations with respect to case parameter
	
stenosis2dsimplemix_0hb.py : full size Mix Network
	
stenosis2dsimplemix_plus_0hb.py : plus size Mix Network
	
stenosis2dsimplemix_plus_0io_0hb.py : plus size Mix Network without tube-specific coordinates input
	
stenosis2dsimplemix_plus_deqn_0hb.py : full size Mix Network with added derivatives of governing and boundary equations with respect to case parameter
	
stenosis2dsimplesingle256_io.py : single PINN Network with 256 nodes per layer (4 layers)
	
stenosis2dsimplesingle384_0io.py : single PINN Network with 384 nodes per layer (4 layers) without tube-specific coordinates input
	
stenosis2dsimplesingle512_0io.py : single PINN Network with 512 nodes per layer (4 layers) without tube-specific coordinates input
	
stenosis2dsimplesingle1024_0io.py : single PINN Network with 1024 nodes per layer (4 layers) without tube-specific coordinates input
