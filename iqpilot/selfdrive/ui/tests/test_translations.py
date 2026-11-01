import gettext
import json
import re

import pytest

from iqpilot.system.ui.lib.multilang import LANGUAGES_FILE, TRANSLATIONS_DIR


FORMAT_ARG = re.compile(r"%(?:\([^)]+\))?[#0+\-]?(?:\d+|\*)?(?:\.\d+|\.\*)?[hlL]?[diouxXeEfFgGcrsa%]")


with LANGUAGES_FILE.open(encoding="utf-8") as stream:
  LANGUAGES = json.load(stream)


def load_catalog(language_code: str) -> dict[str | tuple[str, int], str]:
  with TRANSLATIONS_DIR.joinpath(f"app_{language_code}.mo").open("rb") as stream:
    return gettext.GNUTranslations(stream)._catalog


def message_keys(catalog: dict[str | tuple[str, int], str]) -> set[str]:
  return {key for key in catalog if isinstance(key, str) and key}


def format_args(text: str) -> list[str]:
  return sorted(match for match in FORMAT_ARG.findall(text) if match != "%%")


def test_language_codes_are_unique():
  assert len(LANGUAGES) == len(set(LANGUAGES.values()))


@pytest.mark.parametrize("language_code", LANGUAGES.values(), ids=LANGUAGES.keys())
def test_translation_catalog_is_complete(language_code):
  source = load_catalog("en")
  translated = load_catalog(language_code)
  assert message_keys(translated) == message_keys(source)


@pytest.mark.parametrize("language_code", LANGUAGES.values(), ids=LANGUAGES.keys())
def test_translation_catalog_entries(language_code):
  catalog = load_catalog(language_code)
  for source, translated in catalog.items():
    if not isinstance(source, str) or not source:
      continue
    assert translated
    assert format_args(translated) == format_args(source)
