import cv2
from pathlib import Path
from os import listdir
from os.path import join, isfile

def make_tag(name):
    wdir_pth = Path(__file__).resolve().parents[1]
    nametag_characters_pth = Path(wdir_pth,"templates","nametag_characters")  
    character_t = [cv2.imread(join(nametag_characters_pth, f),0) for f in sorted(listdir(nametag_characters_pth)) if isfile(join(nametag_characters_pth, f))]

    nametag_im = cv2.imread(join(nametag_characters_pth, "space.png"),0)

    for l in name:
        if l != " ":
            if l.isupper():
                char = cv2.imread(join(nametag_characters_pth, f"{l}_.png"),0)
            else:
                char = cv2.imread(join(nametag_characters_pth, f"{l}.png"),0)

            space = cv2.imread(join(nametag_characters_pth, "blank.png"),0)
            new_chars = cv2.hconcat([char,space])
            nametag_im = cv2.hconcat([nametag_im, new_chars])
        else:
            space = cv2.imread(join(nametag_characters_pth, "space.png"),0)
            nametag_im = cv2.hconcat([nametag_im, space])

    return nametag_im


if __name__ == "__main__":
    nametag = make_tag('Pink Bunny')

    cv2.imshow("test", nametag)
    cv2.waitKey()
    cv2.destroyAllWindows()