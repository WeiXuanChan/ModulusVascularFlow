from setuptools import setup
setup(
  name = 'modulusDL',         # How you named your package folder (MyLib)
  packages = ['modulusDL','modulusDL.eq','modulusDL.geometry','modulusDL.loss','modulusDL.models','modulusDL.models.layers','modulusDL.solver'],  
  include_package_data=True,   
  package_data={
        'eq':['*'],
        'geometry':['*'],
        'loss':['*'],
        'models':['*'],
        'solver':['*'],
        },
  version = '1.0.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Codes extention from NVIDIA Modulus.',   # Give a short description about your library
  author = 'Wei Xuan Chan',                   # Type in your name
  author_email = 'w.x.chan1986@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/WeiXuanChan/ModulusVascularFlow',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/WeiXuanChan/ModulusVascularFlow/archive/v1.0.0.tar.gz',    # I explain this later on
  keywords = ['Modulus', 'pytorch', 'PINN'],   # Keywords that define your package best
  install_requires=[],
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package    
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',    
    'License :: OSI Approved :: MIT License',   # Again, pick a license   
    'Programming Language :: Python :: 3.8',
  ],
)
