"""Scene-independent detection vocabulary and validated language task specification."""
import re

# Identical for every scene; never populated from simulator metadata.
COMMON_CATEGORIES = ('chair', 'table', 'sofa', 'bed', 'cabinet', 'refrigerator',
                     'door', 'lamp', 'television', 'monitor', 'sink', 'plant')
ALIASES = {'couch': 'sofa', 'fridge': 'refrigerator', 'tv': 'television',
           'computer monitor': 'monitor', 'potted plant': 'plant', 'dining table': 'table'}


def category_name(value):
    if not isinstance(value, str):
        raise ValueError('Category must be an English object noun')
    value = ' '.join(value.lower().strip().split())
    if not re.fullmatch(r'[a-z]+(?:[ -][a-z]+){0,4}', value):
        raise ValueError('Invalid detection category: '+value)
    return ALIASES.get(value, value)


def validate_task(data):
    if not isinstance(data, dict) or data.get('supported') is not True:
        raise ValueError('Unsupported instruction: '+str(data.get('reason', '') if isinstance(data, dict) else data))
    def names(key, required=False):
        values = data.get(key, [])
        if not isinstance(values, list) or len(values) > 8 or (required and not values):
            raise ValueError('Invalid '+key)
        return list(dict.fromkeys(category_name(x) for x in values))
    target = names('target_categories', True)
    reference = names('reference_categories')
    attributes = data.get('attributes', [])
    relation = data.get('relation', '')
    if not isinstance(attributes, list) or not all(isinstance(x, str) for x in attributes) or not isinstance(relation, str):
        raise ValueError('Invalid task attributes/relation')
    return dict(target_categories=target, reference_categories=reference,
                attributes=attributes, relation=relation,
                detection_categories=list(dict.fromkeys(target+reference+list(COMMON_CATEGORIES))))

# Deterministic grammar for unambiguous single-object requests; shared across scenes.
# Unknown nouns and relational/compound instructions still go to the language parser.
_SIMPLE_NOUNS = {'椅子':'chair','桌子':'table','沙发':'sofa','床':'bed','柜子':'cabinet',
                 '冰箱':'refrigerator','门':'door','灯':'lamp','电视':'television','显示器':'monitor',
                 '水槽':'sink','植物':'plant','微波炉':'microwave'}
_SIMPLE_COLORS = {'红色':'red','白色':'white','棕色':'brown','褐色':'brown','黑色':'black',
                  '灰色':'gray','蓝色':'blue','绿色':'green','黄色':'yellow'}


def simple_navigation_task(instruction):
    text = instruction.strip().rstrip('。！!')
    pattern = (r'(?:请)?(?:走到|走向|前往|去|找到|寻找|靠近)\s*'
               r'(?:(?:任意|任何)(?:一)?(?:个|把|张|台|盏)?)?\s*'
               r'(?P<color>'+ '|'.join(_SIMPLE_COLORS) +r')?\s*(?:的)?'
               r'(?P<noun>'+ '|'.join(_SIMPLE_NOUNS) +r')(?:旁边|附近|边上)?')
    match = re.fullmatch(pattern, text)
    if not match:
        return None
    task = validate_task(dict(supported=True,target_categories=[_SIMPLE_NOUNS[match['noun']]],
                              attributes=[_SIMPLE_COLORS[match['color']]] if match['color'] else []))
    task['parse_method'] = 'single_object_grammar'
    return task
