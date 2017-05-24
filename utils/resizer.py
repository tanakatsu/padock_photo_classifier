from PIL import Image
import sys


def resize(inputfile, outputfile, target_width, target_height):
    img = Image.open(inputfile)

#    if img.width > img.height:
#        x1 = (img.width - img.height) / 2
#        x2 = img.width - x1
#        box = (x1, 0, x2, img.height)
#        img = img.crop(box)
#    else:
#        y1 = (img.height - img.width) / 2
#        y2 = img.height - y1
#        box = (0, y1, img.width, y2)
#        img = img.crop(box)

    resized_img = img.resize((target_width, target_height))
    # resized_img = img.resize((target_width, target_height), resample=Image.LANCZOS)
    resized_img.save(outputfile, 'JPEG')

if __name__ == '__main__':
    print('input=%s, output=%s' % (sys.argv[1], sys.argv[2]))
    resize(sys.argv[1], sys.argv[2], 32, 32)
