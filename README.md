# MCBPixelArtGenerator
Some of the artwork made using this program:

[![Instagram](https://img.shields.io/badge/Subscribe-On_YouTube-E62117?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@Warden_Whisperer)
![YouTube Views](https://img.shields.io/badge/Views-15M+-E62117?style=for-the-badge&logo=youtube&logoColor=white)
<br>
[![Instagram](https://img.shields.io/badge/Follow-On_Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/wardenwhisperer)
![Instagram Views](https://img.shields.io/badge/Views-3M+-E4405F?style=for-the-badge&logo=instagram&logoColor=white)


## Website

COMING SOON! (You'll be able to access all these features + more, without installations and on ANY device)

## Installation
Clone this repo. Inside the repo, run `pip install requirements.txt`

## Usage
Run `python add_to_minecraft.py`

The script will prompt you through an installation process. After it is done you will be left with a `.mcaddon` file. Right click and select `Open With` → `Minecraft Bedrock`.

Now create a new superflat world and under behaviour packs enable the pack (its icon matches the image you used.)

Once you are inside the world, you can start summon the artwork by breaking a grass block. 


## FAQ
- ***How do I set the world type to `superflat`?*** Under create new world → Advanced → Enable `Flat world`
- ***How do I add a behvaiour pack?*** Under world settings → Behaviour Packs → Available → Click `Activate` next to the pack you want to activate.
- ***How can I use this on my phone/tablet?*** Once you have created the `.mcaddon` email it to your phone/tablet. Download the file onto that device. You can then find the file in `Downloads`, `Files` or `File Manager` depending on your device. Double click the file and it should automatically open in Minecraft (ensure Minecraft is installed and updated to the latest version).
- ***Why didn't my artwork get summoned?*** Currently it can only be summoned at spawn (0,-60,0 coordinates).
- ***Only part of the artwork is visible.*** This is a known issue on Minecraft Bedrock edition, especially on lower end devices (phones, tablets, etc.) due to render distance restraints. You must go near the area where the image did not generate correctly, then break a grass block there.


## Are you a developer who wants to contribute?
Feel free to add any of the following features (or anything else) and open a pull request!
- Automatic addition of entire image (even outside of the render distance) using ticking areas or similar. Requires addition and removal of ticking areas, since they are capped at 10.
- JS implementation (particularly of the CEILAB conversion of pixels). Needs to be fast and accurate.
- Website front-end (home page with simple uploading of image and displays. Even a CSS file and cooresponding HTML template will help!!)
- Adding a full 3D model. Use the same technique of creating the behaviour pack `.js`. Implement a way to covnert 3D files into a list of blocks and coords that can be injected into a behaviour pack.
