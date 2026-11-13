"""Render a new preview from historical geometry, without altering call logs."""
import json
from pathlib import Path
from PIL import Image
from visual_nav.annotations import draw_labels


if __name__ == '__main__':
    run = Path('outputs/web_visual_20261106_134404_686145')
    request = json.loads((run/'vlm_requests/request_0004.json').read_text())
    context = json.loads(request['content'][0]['text'])
    image = Image.open(run/(context['observation_id']+'.jpg')).convert('RGB')
    candidates = [dict(id=i['id'], category=i['detected_category'], bbox=i['bbox'])
                  for i in context['visible_candidates']]
    layout = draw_labels(image, candidates, selected=context['candidate_id'])
    output = Path('outputs/label_layout_preview')
    output.mkdir(exist_ok=True)
    image.save(output/'blue_cabinet_new_labels.jpg', quality=95)
    print(json.dumps(layout, indent=2))
