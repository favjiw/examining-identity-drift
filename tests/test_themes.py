import json
import re
from pathlib import Path
from idrift.config import load_config
from idrift.conversation import build_messages, _theme_text


def test_themes_json_structure():
    path = Path("data/themes.json")
    assert path.exists(), "data/themes.json harus ada"
    
    with open(path, encoding="utf-8") as f:
        themes = json.load(f)

    # 1. Tepat 36 entri
    assert len(themes) == 36, "Harus tepat 36 entri"

    ids = [t["id"] for t in themes]
    # 2. ID berurutan 1..36, tidak ada duplikat
    assert ids == list(range(1, 37)), "ID harus 1..36 berurutan"
    assert len(set(ids)) == 36, "Tidak boleh ada ID duplikat"

    for t in themes:
        tid = t["id"]
        tset = t["set"]
        block = t["snapshot_block"]
        text = t["text"].strip()

        # 3. Set dan snapshot_block
        if 1 <= tid <= 12:
            assert tset == "I" and block == 1, f"Tema {tid} harus set I, snapshot_block 1"
        elif 13 <= tid <= 24:
            assert tset == "II" and block == 2, f"Tema {tid} harus set II, snapshot_block 2"
        elif 25 <= tid <= 36:
            assert tset == "III" and block == 3, f"Tema {tid} harus set III, snapshot_block 3"

        # 4. Tidak kosong, tidak mengandung TODO/TODO_VERIFY, tidak diawali angka
        assert len(text) > 0, f"Tema {tid} tidak boleh kosong"
        assert "TODO" not in text and "TODO_VERIFY" not in text, f"Tema {tid} mengandung TODO"
        assert not re.match(r"^\d+[\.\)]", text), f"Tema {tid} tidak boleh diawali angka/nomor"

    # 5. Kata kunci specific tema
    t34 = next(t for t in themes if t["id"] == 34)["text"]
    assert "catches fire" in t34, "Tema 34 harus mengandung 'catches fire'"

    t11 = next(t for t in themes if t["id"] == 11)["text"]
    assert "4 minutes" in t11, "Tema 11 harus mengandung '4 minutes'"

    t22 = next(t for t in themes if t["id"] == 22)["text"]
    assert "total of 5 items" in t22, "Tema 22 harus mengandung 'total of 5 items'"

    t25 = next(t for t in themes if t["id"] == 25)["text"]
    assert ". . ." in t25, "Tema 25 harus mengandung '. . .'"

    t26 = next(t for t in themes if t["id"] == 26)["text"]
    assert ". . ." in t26, "Tema 26 harus mengandung '. . .'"


def test_loader_reads_themes(tmp_path):
    path = Path("data/themes.json")
    with open(path, encoding="utf-8") as f:
        themes = json.load(f)

    # Loader conversation dapat membaca teks tema
    assert _theme_text(themes, 0) == themes[0]["text"]
    assert _theme_text(themes, 33) == themes[33]["text"]

    # Integrasi dengan build_messages
    msgs = build_messages(
        agent="A",
        theme_idx=0,
        history=[],
        themes=themes,
        system_prompt="sys",
    )
    assert themes[0]["text"] in msgs[1]["content"]
