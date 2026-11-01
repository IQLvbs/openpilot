# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import pytest

from iqpilot.selfdrive.assets.fonts import process


@pytest.mark.parametrize('export_succeeds', [True, False])
def test_font_generation_releases_native_allocations(tmp_path, monkeypatch, export_succeeds):
  source = process.FONT_DIR / 'Inter-Regular.ttf'
  monkeypatch.setattr(process, 'FONT_DIR', tmp_path)
  allocations = {}
  freed = set()
  load_font_data = process.rl.load_font_data
  gen_image_font_atlas = process.rl.gen_image_font_atlas
  unload_font_data = process.rl.unload_font_data
  unload_image = process.rl.unload_image
  mem_free = process.rl.mem_free

  def load(*args):
    glyphs = load_font_data(*args)
    count = args[-1][0] if len(args) == 7 else args[4]
    allocations['glyphs'] = glyphs, count
    return glyphs

  def generate(glyphs, rects, *args):
    image = gen_image_font_atlas(glyphs, rects, *args)
    allocations['image'] = image
    allocations['rects'] = rects[0]
    atlas = bytes(process.rl.ffi.buffer(image.data, image.width * image.height * 2))
    glyph_matches = []
    for index in range(allocations['glyphs'][1]):
      glyph = glyphs[index].image
      rect = rects[0][index]
      matches = (rect.width == glyph.width and rect.height == glyph.height
                 and 0 <= rect.x <= image.width - rect.width and 0 <= rect.y <= image.height - rect.height)
      if matches and glyph.width * glyph.height:
        pixels = bytes(process.rl.ffi.buffer(glyph.data, glyph.width * glyph.height))
        for row in range(glyph.height):
          start = ((int(rect.y) + row) * image.width + int(rect.x)) * 2 + 1
          matches &= pixels[row * glyph.width:(row + 1) * glyph.width] == atlas[start:start + glyph.width * 2:2]
      glyph_matches.append(matches)
    allocations['glyph_matches'] = glyph_matches
    return image

  def free_glyphs(*args):
    freed.add('glyphs')
    unload_font_data(*args)

  def free_image(image):
    freed.add('image')
    unload_image(image)

  def free_rects(rects):
    freed.add('rects')
    mem_free(rects)

  monkeypatch.setattr(process.rl, 'load_font_data', load)
  monkeypatch.setattr(process.rl, 'gen_image_font_atlas', generate)
  monkeypatch.setattr(process.rl, 'unload_font_data', free_glyphs)
  monkeypatch.setattr(process.rl, 'unload_image', free_image)
  monkeypatch.setattr(process.rl, 'mem_free', free_rects)
  if not export_succeeds:
    monkeypatch.setattr(process.rl, 'export_image', lambda *args: False)

  try:
    if export_succeeds:
      process._process_font(source, tuple(range(32, 127)))
      assert (tmp_path / 'Inter-Regular.png').is_file()
      assert 'chars count=95' in (tmp_path / 'Inter-Regular.fnt').read_text()
    else:
      with pytest.raises(RuntimeError, match='Failed to export atlas image'):
        process._process_font(source, tuple(range(32, 127)))
    assert freed == {'glyphs', 'rects', 'image'}
    assert all(allocations['glyph_matches'])
  finally:
    if 'glyphs' in allocations and 'glyphs' not in freed:
      unload_font_data(*allocations['glyphs'])
    if 'image' in allocations and 'image' not in freed:
      unload_image(allocations['image'])
    if 'rects' in allocations and 'rects' not in freed:
      mem_free(allocations['rects'])
