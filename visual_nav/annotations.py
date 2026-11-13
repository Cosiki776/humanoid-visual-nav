"""Compact frame-local labels with collision avoidance and leader lines."""
from PIL import ImageDraw, ImageFont


def overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def label_position(box, size, canvas, occupied, objects):
    width, height = size
    cw, ch = canvas
    x1, y1, x2, y2 = box
    positions = [(x1, y1-height-4), (x1, y2+4), (x2+4, y1), (x1-width-4, y1)]
    # Search the entire image if the local options overlap other labels.
    positions += [(x, y) for y in range(0, ch, height+4)
                  for x in (x1, x2-width, 0, cw-width)]
    def rect(pos):
        x, y = max(0, min(pos[0], cw-width)), max(0, min(pos[1], ch-height))
        return (x, y, x+width, y+height)
    def score(r):
        return (sum(overlap(r, other) for other in occupied),
                sum(overlap(r, other) for other in objects),
                abs(r[0]-x1)+abs(r[3]-y1))
    return min(map(rect, positions), key=score)


def draw_labels(image, candidates, selected=None, excluded=()):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    items = [i for i in candidates if i.get('bbox') is not None]
    items.sort(key=lambda i: i['id'] != selected)
    objects = [i['bbox'] for i in items]
    occupied, layouts = [], []
    for item in items:
        color = (255, 210, 0) if item['id'] == selected else (0, 220, 255)
        draw.rectangle(tuple(item['bbox']), outline=color, width=2)
    for item in items:
        box = item['bbox']
        color = (255, 210, 0) if item['id'] == selected else (0, 220, 255)
        label = item['id']+' '+item['category']+(' excluded' if item['id'] in excluded else '')
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        r = label_position(box, (right-left+6, bottom-top+4), image.size, occupied, objects)
        occupied.append(r)
        layouts.append(dict(id=item['id'], label=label, rect=r, color=color, offset=(left, top)))
        cx, cy = (r[0]+r[2])/2, (r[1]+r[3])/2
        anchor = (max(box[0], min(cx, box[2])), max(box[1], min(cy, box[3])))
        start = (max(r[0], min(anchor[0], r[2])), max(r[1], min(anchor[1], r[3])))
        draw.line((start, anchor), fill=color, width=1)
    # Labels last: no subsequent box edge or leader may obscure their text.
    for item in layouts:
        x, y = item['rect'][:2]
        left, top = item['offset']
        draw.rectangle(item['rect'], fill=item['color'])
        draw.text((x+3-left, y+2-top), item['label'], font=font, fill=(0, 0, 0))
    return layouts
