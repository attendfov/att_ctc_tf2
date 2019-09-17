# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io


def write_html(html_file, image_list, label_list, infer_list):
    assert(len(image_list) == len(infer_list))
    assert(len(image_list) == len(label_list))

    index = io.open(html_file, "w", encoding='utf-8')
    index.write("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />")
    index.write("<html><body><table><tr>")
    index.write("<th>name</th><th>image</th><th>label</th><th>infer</th></tr>")

    for image_file, label, infer in zip(image_list, label_list, infer_list):
        if not os.path.isfile(image_file):
            continue

        index.write("<td>%s</td>" % os.path.basename(image_file))
        index.write("<td><img src='%s'></td>" % os.path.abspath(image_file))
        index.write("<td>%s</td>" % str(label))
        index.write("<td>%s</td>" % str(infer))
        index.write("</tr>")
    index.close()
