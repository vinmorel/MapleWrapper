# MapleWrapper

_A MapleStory GMS v.92 and below client wrapper for real-time game data extraction. Enables you to implement your own reinforcement learning environments and bots._

**MapleWrapper:**

* Is incredibly user-friendly. Get started with 2 lines of code. 
* Extracts player position, player stats, and mob position out of the box 
* Is easy to debug. Use the inspect() method to figure out what's going on behind the scenes.
* Is thredded FAST!

### QuickStart Guide
**Instantiate the MapleWrapper**
```python
from maplewrapper import wrapper

w = wrapper('Your_Player_IGN', mobs=['Mob_Name'])
```
You instantiate the wrapper by passing your player in-game name and a list of valid mob names (see [handbook](/maplewrapper/utils/mobs.txt)) that you would like the wrapper to detect. 

**Extract Game info**


```python
from maplewrapper import wrapper

with wrapper('Your_Player_IGN', mobs=['Mob_Name']) as w:
    w.observe(verbose=1)
```
```
Out[1]: 
Player: [394 186 442 202] 
Stats:  [11, 261, 118, 7]
Mobs:   [[354  67 417  97]
         [ 89  67 152  97]]
```
The _observe_ method returns three numpy array objects representing [x0, y0, x1, y1] hitbox coordinates for player and mobs and [LVL, HP, MP, EXP] for stats. The _start_ and _stop_ methods respectively start and stop the game capture. 

**Alternatively, High speed extraction**
```python
from maplewrapper import wrapper

with wrapper('Your_Player_IGN', mobs=['Mob_Name']) as w:
    while True:
        player, stats, mobs = w.observe()
```

**Extract information by category**


```python
from maplewrapper import wrapper

with wrapper('Your_Player_IGN', mobs=['Mob_Name']) as w:
    # Player hitbox coordinates [x0,y1,x1,y1]
    player = w.get_player()
    
    # Player current stats [LVL,HP,MP,EXP]  
    stats = w.get_stats()
    
    # Player current base stats [LVL,Total_HP,Toal_MP,Total_EXP_required_for_LVL]  
    base_stats = w.get_basestats()
    
    # Mobs hitbox coordinates [x0,y0,x1,y1]
    mobs = w.get_mobs()

print(player, stats, base_stats, mobs)
```
```
Out[1]:  
[394 186 442 202] [11, 261, 118, 7] [42, 30000, 30000, 285532] []
```
The _get_basestats_ method returns a numpy array objects representing [LVL, HP, MP, EXP] of your player's current base stats. In other words, you get your players current level, total current possible HP/MP and the total amount of EXP for the LVL. 


**Debug Wrapper**

```python
from maplewrapper import wrapper

w = wrapper('smashy', mobs=['Red Snail'])
w.inspect('mobs')
```
![](/assets/mobs.png) 

The _inspect_ method displays the image crop of what the wrapper sees during data extraction. It will also display an overlay of the bounding box of predictions when applicable.

The required argument ```item``` can take the following values:
* ```'frame'``` : The complete game frame
* ```'content'``` : The cropped game frame containing game content such as the player and the mobs
* ```'ui'``` : The cropped game frame containing stats info
* ```'player'``` : The content frame and player detection overlay 
* ```'mobs'``` : The cropped content frame and mobs detection overlay
* ```'stats'``` : The ui frame and stats detection overlay
* ```'base_stats'``` : The ui frame and base stats detection overlay
* ```'nametag_t'``` : The generated nametag template
* ```'mobs_t'``` : The generated mob templates

You can save to disk by passing optional argument ```save_to_disk=True```


## Requirements
* Windows 7 and up
* Python 3.6+ 


## Installation
**Clone repository**
```
git clone https://github.com/vinmorel/MapleWrapper.git
```
**Install**
```
cd ./MapleWrapper
pip install .
```

### In-Depth Explanations
The beauty of this wrapper comes from the fact that you don't have to do any dirty image manipulations yourself in order to adapt the wrapper to your system and needs. Everything is generated on the fly and cached for future use. 

The MapleWrapper object hooks to your MapleStory game window via [D3DShot](https://github.com/SerpentAI/D3DShot) screen capture. Therefore, your game window must be visible when running wrapper methods. It is recommended that you use the original 800x600 resolution game clients as this wrapper was built on top of it and there may be compatibility issues with other community made client resolutions. 

Most of the information is extracted using OpenCV image processing methods like template matching: 
* The player position is located using a generated nametag template and matched against the game screen. 
* Similarly, the mobs are located using template matching. Sprites are downloaded from the [Maplestory.io](https://maplestory.io/) API and cached on your system for future use. This keeps the repo slim by preventing the push of unecessary image files. 
* As for the player stats, a custom and efficient character recognition system is applied to the game ui.


### Acknowledgement
Thank you to the team at Serpent.AI for the awesome [D3DShot](https://github.com/SerpentAI/D3DShot) package.
