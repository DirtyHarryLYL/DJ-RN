
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

try:
  FONT = ImageFont.truetype('arial.ttf', 24)
except IOError:
  FONT = ImageFont.load_default()

def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font, color='black', thickness=4):
  draw = ImageDraw.Draw(image)
  (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  text_bottom = bottom
  text_width, text_height = font.getsize(display_str)
  margin = np.ceil(0.05 * text_height)
  draw.rectangle(
      [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                        text_bottom)],
      fill=color)
  draw.text(
      (left + margin, text_bottom - text_height - margin),
      display_str,
      fill='black',
      font=font)

  return image

def draw_bounding_boxes(image, gt_boxes, im_info):
  num_boxes = gt_boxes.shape[0]
  gt_boxes_new = gt_boxes.copy()
  gt_boxes_new[:,:4] = np.round(gt_boxes_new[:,:4].copy() / im_info[2])
  disp_image = Image.fromarray(np.uint8(image[0]))

  for i in range(num_boxes):
    this_class = int(gt_boxes_new[i, 4])
    disp_image = _draw_single_box(disp_image, 
                                gt_boxes_new[i, 0],
                                gt_boxes_new[i, 1],
                                gt_boxes_new[i, 2],
                                gt_boxes_new[i, 3],
                                'N%02d-C%02d' % (i, this_class),
                                FONT,
                                color=STANDARD_COLORS[this_class % NUM_COLORS])

  image[0, :] = np.array(disp_image)
  return image

def draw_bounding_boxes_HOI(image, gt_boxes, gt_class):
  num_boxes = gt_boxes.shape[0]
  gt_boxes_new = gt_boxes.copy()
  gt_boxes_new = np.round(gt_boxes_new[:,1:].copy())
  disp_image = Image.fromarray(np.uint8(image[0]))


  show_list = [99,99,99,99,99,99,99,99]
  count = 0
  for idx, val in enumerate(gt_class[0,:]):
    if val != 0:
      show_list[count] = idx
      count += 1

  for i in [0]: 
    this_class = 0
    disp_image = _draw_single_box(disp_image, 
                                gt_boxes_new[i, 0],
                                gt_boxes_new[i, 1],
                                gt_boxes_new[i, 2],
                                gt_boxes_new[i, 3],
                                '%01d-%01d-%01d-%01d-%01d-%01d-%01d-%01d' % (show_list[0], show_list[1], show_list[2], show_list[3], show_list[4], show_list[5], show_list[6], show_list[7]),
                                FONT,
                                color='Red')

  image[0, :] = np.array(disp_image)
  return image

def draw_bounding_boxes_PVP(image, P_boxes, PVP0, PVP1, PVP2, PVP3, PVP4, PVP5):
  num_boxes = P_boxes.shape[0]
  P_boxes_new = P_boxes.copy()
  P_boxes_new = np.round(P_boxes_new[:, :, 1:].copy())
  disp_image = Image.fromarray(np.uint8(image[0]))

  for i in [0]: 
    disp_str = 'Ankle'
    for j in range(PVP0.shape[1]):
      if PVP0[0, j] != 0:
        disp_str += ' ' + str(j)
    disp_image = _draw_single_box(disp_image, 
                                  P_boxes_new[i, 0, 0],
                                  P_boxes_new[i, 0, 1],
                                  P_boxes_new[i, 0, 2],
                                  P_boxes_new[i, 0, 3],
                                  disp_str,
                                  FONT,
                                  color='Red')
    disp_str = 'Knee'
    for j in range(PVP1.shape[1]):
      if PVP1[0, j] != 0:
        disp_str += ' ' + str(j)
    disp_image = _draw_single_box(disp_image, 
                                  P_boxes_new[i, 1, 0],
                                  P_boxes_new[i, 1, 1],
                                  P_boxes_new[i, 1, 2],
                                  P_boxes_new[i, 1, 3],
                                  disp_str,
                                  FONT,
                                  color='Red')
    disp_str = 'Hip'
    for j in range(PVP2.shape[1]):
      if PVP2[0, j] != 0:
        disp_str += ' ' + str(j)
    disp_image = _draw_single_box(disp_image, 
                                  P_boxes_new[i, 4, 0],
                                  P_boxes_new[i, 4, 1],
                                  P_boxes_new[i, 4, 2],
                                  P_boxes_new[i, 4, 3],
                                  disp_str,
                                  FONT,
                                  color='Red')
    disp_str = 'Hand'
    for j in range(PVP3.shape[1]):
      if PVP3[0, j] != 0:
        disp_str += ' ' + str(j)
    disp_image = _draw_single_box(disp_image, 
                                  P_boxes_new[i, 6, 0],
                                  P_boxes_new[i, 6, 1],
                                  P_boxes_new[i, 6, 2],
                                  P_boxes_new[i, 6, 3],
                                  disp_str,
                                  FONT,
                                  color='Red')
    disp_str = 'Shoulder'
    for j in range(PVP4.shape[1]):
      if PVP4[0, j] != 0:
        disp_str += ' ' + str(j)
    disp_image = _draw_single_box(disp_image, 
                                  P_boxes_new[i, 7, 0],
                                  P_boxes_new[i, 7, 1],
                                  P_boxes_new[i, 7, 2],
                                  P_boxes_new[i, 7, 3],
                                  disp_str,
                                  FONT,
                                  color='Red')
    disp_str = 'Head'
    for j in range(PVP5.shape[1]):
      if PVP5[0, j] != 0:
        disp_str += ' ' + str(j)
    disp_image = _draw_single_box(disp_image, 
                                  P_boxes_new[i, 5, 0],
                                  P_boxes_new[i, 5, 1],
                                  P_boxes_new[i, 5, 2],
                                  P_boxes_new[i, 5, 3],
                                  disp_str,
                                  FONT,
                                  color='Red')

  image[0, :] = np.array(disp_image)
  return image

def draw_bounding_boxes_HOI_PIC(image, gt_boxes, gt_class):
  num_boxes = gt_boxes.shape[0]
  gt_boxes_new = gt_boxes.copy()
  gt_boxes_new = np.round(gt_boxes_new[:,1:].copy())
  disp_image = Image.fromarray(np.uint8(image[0]))


  show_list = [99,99,99,99,99,99,99,99]
  count = 0
  for idx, val in enumerate(gt_class[0,:]):
    if val != 0:
      show_list[count] = idx
      count += 1
  show_list[1] = gt_boxes_new[2, 0]
  show_list[2] = gt_boxes_new[2, 1]
  show_list[3] = gt_boxes_new[2, 2]
  show_list[4] = gt_boxes_new[2, 3]


  for i in [0]:
    this_class = 0
    disp_image = _draw_single_box(disp_image, 
                                gt_boxes_new[i, 0],
                                gt_boxes_new[i, 1],
                                gt_boxes_new[i, 2],
                                gt_boxes_new[i, 3],
                                '%01d-%01d-%01d-%01d-%01d-%01d-%01d-%01d' % (show_list[0], show_list[1], show_list[2], show_list[3], show_list[4], show_list[5], show_list[6], show_list[7]),
                                FONT,
                                color='Red')

  image[0, :] = np.array(disp_image)
  return image