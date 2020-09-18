# MapleWrapper

_A MapleStory GMS v.92 and below client wrapper for real-time game data extraction. Enables **you** to implement your own reinforcement learning environments._

**MapleWrapper:**

* Is incredibly user-friendly. Get started with 2 lines of code. 
* Extracts player position, player stats, and mob position out of the box 
* Is easy to debug. Use the investigate() method to figure out what's going on behind the scene.
* Is thredded FAST!

### QuickStart Guide
**Instanciate the MapleWrapper**
```python
from wrapper import MapleWrapper

w = MapleWrapper('Your_Player_IGN', mobs=['Mob_Name'])
```
You instanciate the wrapper by passing your player in-game name and a list of valid mob names (see [handbook](/wrapper/utils/mobs.txt)) that you would like to detect. 

**Extract Game info**


```python
from wrapper import MapleWrapper

w = MapleWrapper('Your_Player_IGN', mobs=['Mob_Name'])

w.start()
w.observe(verbose=1)
w.stop()
```
```
Out[1]: 
Player: [394 186 442 202] 
Stats:  [11, 261, 118, 7]
Mobs:   [[354  67 417  97]
         [ 89  67 152  97]]
```
The _observe_ method returns three numpy array objects representing [x0, y0, x1, y1] coordinates for player and mobs and [LVL, HP, MP, EXP] for stats.  

Alternatively,
```python
from wrapper import MapleWrapper

w = MapleWrapper('Your_Player_IGN', mobs=['Mob_Name'])

w.start()
player, stats, mobs = w.observe()
w.stop()
```

**Debug Wrapper**

```python
from wrapper import MapleWrapper

w = MapleWrapper('smashy', mobs=['Blue Snail'])
w.inspect('mobs')
```
![](/assets/mobs.png) 

The _investigate_ method displays the image crop of what the wrapper sees during data extraction. It will also display an overlay of the bounding box of preditions when applicable.

The required argument ```item``` can take the following values:
* ```'frame'``` : The complete game frame
* ```'content'``` : The cropped game grame containg game content such as the player and the mobs
* ```'ui'``` : The cropped game frame contained stats info
* ```'player'``` : The content frame and player detection overlay 
* ```'mobs'``` : The cropped content frame and mobs detection overlay
* ```'stats'``` : The ui frame and stats detection overlay


## Requirements
* Windows 7 and up
* Python 3.6+ 


## Installation
**Clone repository**
```
git clone https://github.com/vinmorel/MapleWrapper.git
```
**Install requirements**
```
cd ./MapleWrapper
pip install -r requirements.txt
```
***Note: A setup.py will be included in the near future for easier imports***

### In-Depth Explanations
The beauty of this wrapper comes from the fact that you don't have to do any dirty image manipulations yourself in order to adapt the wrapper to your system and needs. Everything is generated on the fly and cached for future use. 

The MapleWrapper object hooks to your MapleStory game window via [D3DShot](https://github.com/SerpentAI/D3DShot) screen capture. Therefore, your game window must be visible when running wrapper methods. It is recommended that you use the original 800x600 resolution game clients as this wrapper was built on top of it and there may be compatibility issues with other community made client resolutions. 

Most of the information is extracted using OpenCV image processing methods like template matching: 
* The player position is located using a generated nametag template and matched against the game screen. 
* Similarly, the mobs are located using template matching. Sprites are downloaded from the [Maplestory.io](https://maplestory.io/) API and cached on your system for future use. This keeps the repo slim by preventing the push of unecessary image files. 
* As for the player stats, a custom and efficient character recognition system is applied to the game ui.
