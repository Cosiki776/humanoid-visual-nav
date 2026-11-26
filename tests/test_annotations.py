import unittest
from PIL import Image
from visual_nav.annotations import draw_labels, overlap


class AnnotationTests(unittest.TestCase):
    def test_adjacent_labels_do_not_overlap(self):
        items = [dict(id='v0027', category='cabinet', bbox=[267,287,439,366]),
                 dict(id='v0028', category='cabinet', bbox=[226,280,274,353]),
                 dict(id='v0029', category='cabinet', bbox=[624,312,654,360])]
        layouts = draw_labels(Image.new('RGB', (768,512)), items, selected='v0027')
        for n, item in enumerate(layouts):
            for other in layouts[n+1:]:
                self.assertEqual(overlap(item['rect'], other['rect']), 0)
            for obj in items:
                self.assertEqual(overlap(item['rect'], obj['bbox']), 0)
        self.assertEqual(layouts[0]['id'], 'v0027')

    def test_edges_and_excluded_label(self):
        layouts = draw_labels(Image.new('RGB', (768,512)), [
            dict(id='v0001', category='chair', bbox=[0,0,40,60]),
            dict(id='v0002', category='chair', bbox=[740,480,767,511])], excluded=['v0001'])
        self.assertIn('excluded', layouts[0]['label'])
        for item in layouts:
            x,y,r,b = item['rect']
            self.assertTrue(0 <= x < r <= 768 and 0 <= y < b <= 512)


if __name__ == '__main__':
    unittest.main()
