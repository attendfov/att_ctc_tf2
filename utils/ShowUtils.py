#-*- coding:utf-8 -*-
import os
import cv2
import copy

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

abspath = os.path.dirname(os.path.realpath(__file__))
abspath = os.path.abspath(abspath)


def PIL2CV(pil_image):
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def CV2PIL(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR))


def show_image(img_url, ttf_url, gt_text, pr_text, imgw_scale=2):
    assert(os.path.isfile(img_url))
    assert(os.path.isfile(ttf_url))
    image = Image.open(img_url)
    imgw, imgh = image.size

    pd_image = np.zeros([imgh + 10, int(imgw * imgw_scale), 3], dtype=np.uint8)
    gt_image = np.zeros([imgh + 10, int(imgw * imgw_scale), 3], dtype=np.uint8)
    pr_image = np.zeros([imgh + 10, int(imgw * imgw_scale), 3], dtype=np.uint8)

    txt_size = int(max(10, min(imgh, imgh)*0.8))
    gt_image = Image.fromarray(gt_image)
    pr_image = Image.fromarray(pr_image)

    font_unic = ImageFont.truetype(ttf_url, txt_size, encoding='unic')

    gt_drawobj = ImageDraw.Draw(gt_image)
    pr_drawobj = ImageDraw.Draw(pr_image)

    gt_drawobj.text((0, 0), str(gt_text), font=font_unic, fill=(0, 0, 0))
    pr_drawobj.text((0, 0), str(pr_text), font=font_unic, fill=(0, 0, 0))

    ori_image = PIL2CV(image)
    grt_image = PIL2CV(gt_image)
    inf_image = PIL2CV(pr_image)

    ori_imgh, ori_imgw = ori_image.shape[:2]
    grt_imgh, grt_imgw = grt_image.shape[:2]
    inf_imgh, inf_imgw = inf_image.shape[:2]

    pd_image[5:ori_imgh+5, :ori_imgw, :] = ori_image

    assert (ori_imgw * imgw_scale == grt_imgw)
    assert (ori_imgw * imgw_scale == inf_imgw)

    merge_img = np.concatenate((grt_image, ori_image, inf_image), axis=0)

    return merge_img


def show_attention_image(img_url,
                         ttf_url,
                         gt_text,
                         pr_text,
                         att_position,
                         imgw_scale=2):
    assert(os.path.isfile(img_url))
    assert(os.path.isfile(ttf_url))
    image = Image.open(img_url)
    imgw, imgh = image.size

    mk_image = np.zeros([imgh + 6, int(imgw * imgw_scale), 3], dtype=np.uint8)
    pd_image = np.zeros([imgh + 6, int(imgw * imgw_scale), 3], dtype=np.uint8)
    gt_image = np.zeros([imgh + 6, int(imgw * imgw_scale), 3], dtype=np.uint8)
    pr_image = np.zeros([imgh + 6, int(imgw * imgw_scale), 3], dtype=np.uint8)

    txt_size = int(max(10, min(imgh, imgh)*0.8))
    mk_image = Image.fromarray(mk_image)
    gt_image = Image.fromarray(gt_image)
    pr_image = Image.fromarray(pr_image)

    font_unic = ImageFont.truetype(ttf_url, txt_size, encoding='unic')

    gt_drawobj = ImageDraw.Draw(gt_image)
    pr_drawobj = ImageDraw.Draw(pr_image)

    if gt_text is not None and len(gt_text) > 0:
        gt_drawobj.text((0, 4), str(gt_text), font=font_unic, fill=(0, 255, 0))

    if pr_text is not None and len(pr_text) > 0:
        pr_drawobj.text((0, 4), str(pr_text), font=font_unic, fill=(0, 0, 255))

    ori_image = PIL2CV(image)
    grt_image = PIL2CV(gt_image)
    inf_image = PIL2CV(pr_image)

    ori_imgh, ori_imgw = ori_image.shape[:2]
    grt_imgh, grt_imgw = grt_image.shape[:2]
    inf_imgh, inf_imgw = inf_image.shape[:2]

    pd_image[5:ori_imgh+5, :ori_imgw, :] = ori_image

    assert (ori_imgw * imgw_scale == grt_imgw)
    assert (ori_imgw * imgw_scale == inf_imgw)

    assert(len(pr_text) == len(att_position))
    merge_imgs = [grt_image, pd_image, inf_image]
    for txt_char, char_pos in zip(pr_text, att_position):
        images = copy.deepcopy(pd_image)
        images = np.array(images, dtype=np.int32)
        for pos in char_pos:
            if pos >= ori_imgw:
                continue
            images[:, pos, :] = images[:, pos, :] + 100
        image1 = np.zeros(images.shape, dtype=np.uint8)
        image1.fill(255)
        imaget = np.where(images > 255, image1, images)
        merge_imgs.append(np.array(imaget, dtype=np.uint8))
        char_image = copy.deepcopy(mk_image)
        pr_drawobj = ImageDraw.Draw(char_image)
        pr_drawobj.text((0, 0), str(txt_char), font=font_unic, fill=(0, 255, 255))
        char_image = PIL2CV(char_image)
        merge_imgs.append(np.array(char_image, dtype=np.uint8))
    merge_img = np.concatenate(merge_imgs, axis=0)
    return merge_img


def show_2dattention_image(img_url,
                           ttf_url,
                           gt_text,
                           pr_text,
                           att_position,
                           imgw_scale=2):
    assert(os.path.isfile(img_url))
    assert(os.path.isfile(ttf_url))
    image = Image.open(img_url)
    imgw, imgh = image.size

    mk_image = np.zeros([imgh + 6, int(imgw * imgw_scale), 3], dtype=np.uint8)
    pd_image = np.zeros([imgh + 6, int(imgw * imgw_scale), 3], dtype=np.uint8)
    gt_image = np.zeros([imgh + 6, int(imgw * imgw_scale), 3], dtype=np.uint8)
    pr_image = np.zeros([imgh + 6, int(imgw * imgw_scale), 3], dtype=np.uint8)

    txt_size = int(max(10, min(imgh, imgh)*0.8))
    mk_image = Image.fromarray(mk_image)
    gt_image = Image.fromarray(gt_image)
    pr_image = Image.fromarray(pr_image)

    font_unic = ImageFont.truetype(ttf_url, txt_size, encoding='unic')

    gt_drawobj = ImageDraw.Draw(gt_image)
    pr_drawobj = ImageDraw.Draw(pr_image)

    if gt_text is not None and len(gt_text) > 0:
        gt_drawobj.text((0, 4), str(gt_text), font=font_unic, fill=(0, 255, 0))

    if pr_text is not None and len(pr_text) > 0:
        pr_drawobj.text((0, 4), str(pr_text), font=font_unic, fill=(0, 0, 255))

    ori_image = PIL2CV(image)
    grt_image = PIL2CV(gt_image)
    inf_image = PIL2CV(pr_image)

    ori_imgh, ori_imgw = ori_image.shape[:2]
    grt_imgh, grt_imgw = grt_image.shape[:2]
    inf_imgh, inf_imgw = inf_image.shape[:2]

    pd_image[5:ori_imgh+5, :ori_imgw, :] = ori_image

    assert (ori_imgw * imgw_scale == grt_imgw)
    assert (ori_imgw * imgw_scale == inf_imgw)

    assert(len(pr_text) == len(att_position))
    merge_imgs = [grt_image, pd_image, inf_image]
    for txt_char, char_pos in zip(pr_text, att_position):
        images = copy.deepcopy(pd_image)
        images = np.array(images, dtype=np.int32)
        for point in char_pos:
            pointx, pointy = point
            if pointx >= ori_imgw or pointy >= ori_imgh:
                continue
            images[pointy][pointx] = images[pointy][pointx] + 100

        image1 = np.zeros(images.shape, dtype=np.uint8)
        image1.fill(255)
        imaget = np.where(images > 255, image1, images)
        merge_imgs.append(np.array(imaget, dtype=np.uint8))
        char_image = copy.deepcopy(mk_image)
        pr_drawobj = ImageDraw.Draw(char_image)
        pr_drawobj.text((0, 0), str(txt_char), font=font_unic, fill=(0, 255, 255))
        char_image = PIL2CV(char_image)
        merge_imgs.append(np.array(char_image, dtype=np.uint8))
    merge_img = np.concatenate(merge_imgs, axis=0)
    return merge_img




if __name__=='__main__':
    img_url = 'default_show/374_UNDERSELLS_82215.jpg.0.0.139.31.jpg'
    ttf_url = '/Users/junhuang.hj/Desktop/code_paper/code/data_gene/china_font/simhei.ttf'
    gt_text = 'undersells'
    pr_text = 'und'
    att_position = [[1,2,3], [4,5,6], [10,11,12]]
    image = show_attention_image(img_url, ttf_url, gt_text, pr_text, att_position, imgw_scale=2)
    cv2.imshow("image", image)
    cv2.waitKey(0)












