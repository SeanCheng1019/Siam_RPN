
# for x in positive_anchor_pos:
#     res = response[x:x + 1, :, :].cpu().detach().numpy()
#     res_show.append(res)

from PIL import Image
import math
img1 = Image.open('/home/csy/Pictures/img.jpg')#图片1
img2 = Image.open('/home/csy/Pictures/res4.jpg')#图片2
w = img1.size[0]
h = img1.size[1]
img2 = img2.resize((w,h))
img1 = img1.convert('RGBA')
img2 = img2.convert('RGBA')
img = Image.blend(img2, img1, 0.5)
img.show()
# img.save( "blend.png")

