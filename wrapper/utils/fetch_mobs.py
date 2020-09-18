import os
import pathlib
from os.path import join
import requests
import zipfile
import win32gui

def process_coords(c_name):
    """
    Returns tuple (x0, y0, x1, y0) of process position on screen
    """
    hdl = win32gui.FindWindow(c_name, None)
    rect = win32gui.GetWindowRect(hdl)
    return rect

class_names = []
def callback(hwnd, extra):
    cname = win32gui.GetClassName(hwnd)
    class_names.append(cname)

def get_classname(search_term):
    win32gui.EnumWindows(callback, None)
    cname = [i for i in class_names if search_term in i]
    return cname[0]
class Invalid_Mob_Name(Exception):
    pass

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_sprites(mob_name):
    """
    Downloads sprites from maplestory.io if not already on disk. 
    Returns : list of downloaded file names
    """
    pth = pathlib.Path(__file__).resolve().parents[1]
    
    output_dir = join(pth, "templates","mobs", mob_name)
    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        with open(join(pth, "utils", "mobs.txt"), "r") as f:
            l = f.readlines() 
            identifiers = [((" ".join(item.split(" ")[2:])).strip(), item.split(" ")[0]) for item in l if "?" not in (" ".join(item.split(" ")[2:]))]
            identifiers = dict(identifiers)

            try:
                mob_id = identifiers[mob_name]
                os.makedirs(output_dir)
            except KeyError:
                raise Invalid_Mob_Name(f"Please verify '{mob_name}' spelling and confirm it is a valid entry at: https://github.com/ronancpl/HeavenMS/blob/master/handbook/Mob.txt")

            api = f"https://maplestory.io/api/GMS/83/mob/{mob_id}/download"

            save_dir = join(output_dir, f"{mob_name}.zip")
            download_url(api, save_dir, chunk_size=128)

            with zipfile.ZipFile(save_dir, 'r') as zip:
                files_in_zip = zip.namelist()
                stances = {
                    "hit" : 1,
                    "stand" : 3,
                    "move" : 3,
                    "attack" : None,
                    "jump" : 1,
                    "fly" : 3
                }

                chosen_templates = []
                for stance, index in stances.items():
                    if stance == "attack":
                        num_attacks = len([f for f in files_in_zip if stance in f])
                        index = num_attacks // 2
                        stance_templates = sorted([f for f in files_in_zip if stance in f])[index:index+1]
                    else:
                        stance_templates = sorted([f for f in files_in_zip if stance in f])[:index]
                    
                    for candidates in stance_templates:
                        chosen_templates.append(candidates)

                for f in chosen_templates:
                    zip.extract(f, join(pth,"templates","mobs",f"{mob_name}"))

                print(f"'{mob_name}' sprites successfully added to cache")

            os.remove(save_dir)

        return chosen_templates


if __name__ == "__main__":
    print(download_sprites("Squid"))